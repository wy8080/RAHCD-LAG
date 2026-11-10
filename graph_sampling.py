"""Random-walk based subgraph sampling for large graphs.

`GraphStorage` keeps the whole graph in CSR form, `weighted_random_walk_partition`
performs feature/topology weighted multi-root random walks and returns node
batches, and `build_batches` materializes each batch (node ids, features,
induced edges).
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from tqdm import tqdm


class GraphStorage:
    """CSR graph storage with a cheap induced-subgraph extractor."""

    def __init__(self, edge_index, num_nodes, x_input, device="cpu"):
        self.num_nodes = num_nodes
        self.device = device
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        self.A = csr_matrix(
            (np.ones(len(row), dtype=np.int8), (row, col)),
            shape=(num_nodes, num_nodes),
        )
        self.A.sort_indices()

        self.indptr = self.A.indptr.astype(np.int64)
        self.indices = self.A.indices.astype(np.int32)
        self.A.data = None
        self.A = None

        self.deg = np.diff(self.indptr).astype(np.float32)
        self.global_m2 = float(self.deg.sum())
        self._x_input = x_input

    def extract_subgraph(self, node_ids):
        node_ids_np = node_ids.cpu().numpy()
        n = len(node_ids_np)
        local_id = -np.ones(self.num_nodes, dtype=np.int32)
        local_id[node_ids_np] = np.arange(n)

        sub_rows, sub_cols = [], []
        for i, v in enumerate(node_ids_np):
            s, e = self.indptr[v], self.indptr[v + 1]
            nbrs = self.indices[s:e]
            mask = local_id[nbrs] >= 0
            if mask.any():
                sub_rows.append(np.full(int(mask.sum()), i, dtype=np.int64))
                sub_cols.append(local_id[nbrs[mask]].astype(np.int64))

        if not sub_rows:
            return torch.zeros((2, 0), dtype=torch.long)
        sub_rows = np.concatenate(sub_rows)
        sub_cols = np.concatenate(sub_cols)
        return torch.from_numpy(np.stack([sub_rows, sub_cols]))

    def release(self):
        self._x_input = None
        self.indices = None
        self.indptr = None
        self.deg = None
        gc.collect()


@torch.no_grad()
def weighted_random_walk_partition(
    graph,
    roots_per_batch=2048,
    num_batches=100,
    t_per_root=20,
    walk_depth=5,
    tau_feat=0.9,
    max_batch=50000,
    min_batch=500,
    seed=1,
):
    """Sample node batches by weighted random walks.

    Edge weights combine a feature-similarity term exp(cos / tau_feat) and a
    topology term max(1 - deg_u * deg_v / 2m, 0) + 0.01. Nodes visited more
    often than the batch mean become the batch, then it is capped/clipped.
    """
    num_nodes = graph.num_nodes
    E = len(graph.indices)

    row_id = np.repeat(np.arange(num_nodes), np.diff(graph.indptr))
    CHUNK = 2_000_000
    w_feat = np.zeros(E, dtype=np.float32)

    for start in range(0, E, CHUNK):
        end = min(start + CHUNK, E)
        idx_r = torch.from_numpy(row_id[start:end]).long()
        idx_c = torch.from_numpy(graph.indices[start:end]).long()
        x_r = F.normalize(graph._x_input[idx_r].float(), p=2, dim=1).numpy()
        x_c = F.normalize(graph._x_input[idx_c].float(), p=2, dim=1).numpy()
        cos_sim = np.einsum("ij,ij->i", x_r, x_c)
        w_feat[start:end] = np.exp(cos_sim / tau_feat)
        del x_r, x_c, cos_sim, idx_r, idx_c

    w_topo = np.zeros(E, dtype=np.float32)
    for start in range(0, E, CHUNK):
        end = min(start + CHUNK, E)
        deg_r = graph.deg[row_id[start:end]]
        deg_c = graph.deg[graph.indices[start:end]]
        w_topo[start:end] = (
            np.maximum(1.0 - deg_r * deg_c / graph.global_m2, 0.0) + 0.01
        )
        del deg_r, deg_c
    del row_id

    w_all = np.clip(w_topo * w_feat, 1e-10, None).astype(np.float32)
    del w_topo, w_feat

    row_sums = np.add.reduceat(w_all, graph.indptr[:-1])
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    row_id = np.repeat(np.arange(num_nodes), np.diff(graph.indptr))
    w_norm = (w_all / row_sums[row_id]).astype(np.float32)
    del w_all, row_id, row_sums

    try:
        from numba import njit

        @njit(cache=True)
        def _segment_cumsum(data, indptr, num_nodes):
            out = np.zeros_like(data)
            for i in range(num_nodes):
                s = indptr[i]
                e = indptr[i + 1]
                if e > s:
                    acc = 0.0
                    for j in range(s, e):
                        acc += data[j]
                        out[j] = acc
                    out[e - 1] = 1.0
            return out

        cumsum_data = _segment_cumsum(w_norm, graph.indptr, num_nodes)
    except ImportError:
        cumsum_data = np.zeros(E, dtype=np.float32)
        for i in range(num_nodes):
            s, e = graph.indptr[i], graph.indptr[i + 1]
            if e > s:
                c = np.cumsum(w_norm[s:e])
                c[-1] = 1.0
                cumsum_data[s:e] = c
    del w_norm

    rng = np.random.default_rng(seed)
    batches = []
    for batch_i in range(num_batches):
        roots = rng.choice(num_nodes, roots_per_batch, replace=False)
        batch_visit = np.zeros(num_nodes, dtype=np.int32)
        for root in roots:
            for _ in range(t_per_root):
                cur = int(root)
                for _ in range(walk_depth):
                    s, e = graph.indptr[cur], graph.indptr[cur + 1]
                    if e == s:
                        break
                    c = cumsum_data[s:e]
                    nxt_idx = int(np.searchsorted(c, rng.random(), side="right"))
                    if nxt_idx >= e - s:
                        nxt_idx = e - s - 1
                    cur = int(graph.indices[s + nxt_idx])
                    batch_visit[cur] += 1
        counts = batch_visit[batch_visit > 0]
        if len(counts) == 0:
            del batch_visit, counts
            continue
        mean_count = counts.mean()
        batch = np.where(batch_visit > mean_count)[0]
        if len(batch) > max_batch:
            batch = rng.choice(batch, max_batch, replace=False)
        if len(batch) < min_batch:
            del batch_visit, counts, batch
            continue
        batches.append(torch.from_numpy(batch).long())
        del batch_visit, counts, batch

    del cumsum_data
    gc.collect()

    if len(batches) == 0:
        raise RuntimeError("未生成任何 batch")
    return batches


def build_batches(graph_storage, batches, x_input, desc="Preparing batches"):
    """Materialize sampled node batches into training-ready dicts."""
    batch_data = []
    for b_ids in tqdm(batches, desc=desc):
        sub_ei = graph_storage.extract_subgraph(b_ids)
        batch_data.append(
            {
                "node_ids": b_ids,
                "x": x_input[b_ids],
                "edge_index": sub_ei,
                "num_nodes": len(b_ids),
            }
        )
    return batch_data
#（注：内容由AI生成）

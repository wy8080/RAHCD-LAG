"""Evaluation: modularity statistics and clustering metrics.

Provides:
  * `set_graph_for_modularity` - caches global degree statistics for the whole
    graph (used by the modularity based losses)
  * `modularity_score` - Newman modularity of a hard partition
  * `clustering_metrics` - ARI / NMI / Hungarian-matched ACC
  * `predict_assignments` - hard assignments from a trained model (full graph
    or sampled batches)
  * `evaluate_model` - the above combined into one report

Run as a script to evaluate a checkpoint:

    python evaluation.py --dataset-name ogbn-products \\
        --checkpoint ./checkpoints/ogbn-products/final.pt \\
        --is-large-graph
"""

import argparse
import json
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

_GLOBAL_EDGE_INDEX = None
_GLOBAL_NUM_NODES = None
_GLOBAL_DEG = None
_GLOBAL_M2 = None

if __name__ == "__main__":
    # When executed as a script, let `import evaluation` resolve to this module,
    # so the cached modularity statistics are shared with data_preprocess.
    sys.modules.setdefault("evaluation", sys.modules["__main__"])


def set_graph_for_modularity(edge_index, num_nodes):
    """Cache the global degree vector and 2m for modularity computations."""
    global _GLOBAL_EDGE_INDEX, _GLOBAL_NUM_NODES, _GLOBAL_DEG, _GLOBAL_M2
    ei = edge_index.cpu() if torch.is_tensor(edge_index) else torch.tensor(edge_index)
    if ei.dtype != torch.int32:
        ei = ei.to(torch.int32)
    _GLOBAL_EDGE_INDEX = ei
    _GLOBAL_NUM_NODES = num_nodes
    _GLOBAL_DEG = (
        torch.bincount(ei[0].long(), minlength=num_nodes).numpy().astype(np.float64)
    )
    _GLOBAL_M2 = _GLOBAL_DEG.sum()


def get_global_stats():
    """Return the cached (deg, 2m) or (None, None) when the graph is not set."""
    return _GLOBAL_DEG, _GLOBAL_M2


def modularity_score(edge_index, assignment, num_nodes, global_deg=None, global_m2=None):
    """Newman modularity Q = sum_e delta(c_u, c_v)/2m - sum_k (vol_k/2m)^2."""
    if global_deg is None:
        if _GLOBAL_NUM_NODES == num_nodes and _GLOBAL_DEG is not None:
            global_deg = torch.from_numpy(_GLOBAL_DEG)
            global_m2 = float(_GLOBAL_M2)
        else:
            global_deg = torch.bincount(
                edge_index[0].long(), minlength=num_nodes
            ).double()
            global_m2 = float(global_deg.sum())

    if global_m2 is None:
        global_m2 = float(global_deg.sum())

    deg = torch.as_tensor(global_deg).double()
    assign = torch.as_tensor(assignment).long()
    # global_m2 = total degree over stored (symmetric) edges, i.e. the usual 2m
    m2 = max(float(global_m2), 1.0)

    same = (assign[edge_index[0].long()] == assign[edge_index[1].long()]).double()
    term1 = same.sum() / m2

    K = int(assign.max()) + 1 if assign.numel() > 0 else 0
    vol = torch.zeros(K).double().scatter_add_(0, assign, deg)
    term2 = ((vol / m2) ** 2).sum()

    return float(term1 - term2)


def clustering_metrics(y_true, assignment, num_classes=None):
    """ARI / NMI / ACC against ground-truth labels; unassigned nodes (-1) are dropped."""
    y = np.asarray(y_true).ravel()
    a = np.asarray(assignment).ravel()
    keep = a >= 0
    y, a = y[keep], a[keep]

    result = {
        "n_evaluated": int(keep.sum()),
        "ARI": float(adjusted_rand_score(y, a)),
        "NMI": float(normalized_mutual_info_score(y, a)),
    }

    if y.size == 0:
        result["ACC"] = 0.0
        return result

    K = int(num_classes) if num_classes else int(max(y.max(), a.max())) + 1
    cost = np.zeros((K, K), dtype=np.float64)
    np.add.at(cost, (y.astype(np.int64), a.astype(np.int64)), 1.0)
    rows, cols = linear_sum_assignment(-cost)
    result["ACC"] = float(cost[rows, cols].sum() / y.size)
    return result


@torch.no_grad()
def predict_assignments(model, x_input=None, batch_data=None, num_nodes=None, device="cpu"):
    """Hard assignments (argmax of q) for the full graph or for sampled batches."""
    model.eval()
    if batch_data is None:
        z = model(x_input.to(device))
        q, _ = model.compute_q(z)
        return q.argmax(dim=1).cpu()

    assign = torch.full((num_nodes,), -1, dtype=torch.long)
    for b in batch_data:
        z = model(b["x"].to(device))
        q, _ = model.compute_q(z)
        assign[b["node_ids"]] = q.argmax(dim=1).cpu()
    return assign


@torch.no_grad()
def evaluate_model(
    model,
    y_true,
    num_classes=None,
    x_input=None,
    edge_index=None,
    num_nodes=None,
    batch_data=None,
    device="cpu",
):
    """Compute clustering metrics and modularity for a trained model."""
    assignment = predict_assignments(
        model,
        x_input=x_input,
        batch_data=batch_data,
        num_nodes=num_nodes,
        device=device,
    )
    report = clustering_metrics(y_true, assignment.numpy(), num_classes=num_classes)
    if edge_index is not None:
        report["Modularity"] = modularity_score(edge_index, assignment, num_nodes)
    return report


def _infer_architecture(state_dict):
    """Recover (in_features, middle_dim, hidden_dim, num_clusters, use_ema) from weights."""
    middle_dim, in_features = state_dict["encoder.0.weight"].shape
    hidden_dim = state_dict["encoder.4.weight"].shape[0]
    if "prototype_tangent" in state_dict:
        num_clusters = state_dict["prototype_tangent"].shape[0]
        use_ema = True
    else:
        num_clusters = state_dict["tangent_centers"].shape[0]
        use_ema = False
    return in_features, middle_dim, hidden_dim, num_clusters, use_ema


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--dataset-name", type=str, default="ogbn-products")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--is-large-graph", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k-steps", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--drop-out", type=float, default=0.1)
    parser.add_argument("--ema-momentum", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--roots-per-batch", type=int, default=4096)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--t-per-root", type=int, default=30)
    parser.add_argument("--walk-depth", type=int, default=4)
    parser.add_argument("--tau-feat", type=float, default=0.9)
    parser.add_argument("--max-batch", type=int, default=100000)
    parser.add_argument("--min-batch", type=int, default=1000)
    args = parser.parse_args()

    from data_preprocess import prepare_data
    from graph_sampling import GraphStorage, build_batches, weighted_random_walk_partition
    from hyperbolic import HyperbolicPrototypeEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or f"{args.save_dir}/{args.dataset_name}/final.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_features, middle_dim, hidden_dim, num_clusters, use_ema = _infer_architecture(
        ckpt["model_state_dict"]
    )

    data, _, y_true, num_nodes, _, _ = prepare_data(
        args.dataset_name, args.data_root, k_steps=args.k_steps, alpha=args.alpha
    )

    model = HyperbolicPrototypeEncoder(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_clusters=num_clusters,
        middle_dim=middle_dim,
        drop_out=args.drop_out,
        use_ema_prototype=use_ema,
        ema_momentum=args.ema_momentum,
        is_large_graph=args.is_large_graph,
        seed=args.seed,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    batch_data = None
    if args.is_large_graph:
        graph_storage = GraphStorage(data.edge_index, num_nodes, data.x_input, device)
        batches = weighted_random_walk_partition(
            graph_storage,
            roots_per_batch=args.roots_per_batch,
            num_batches=args.num_batches,
            t_per_root=args.t_per_root,
            walk_depth=args.walk_depth,
            tau_feat=args.tau_feat,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            seed=args.seed,
        )
        batch_data = build_batches(graph_storage, batches, data.x_input)
        graph_storage.release()

    report = evaluate_model(
        model,
        y_true,
        num_classes=num_clusters,
        x_input=None if batch_data is not None else data.x_input,
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        batch_data=batch_data,
        device=device,
    )
    report = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

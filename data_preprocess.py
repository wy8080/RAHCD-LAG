"""Data loading and feature preprocessing.

Covers:
  * loading of the supported datasets (OGB, MAG-Scholar-C, Amazon, Planetoid)
  * label extraction and edge-index dtype normalization
  * degree statistics used by the modularity based losses
  * feature smoothing (propagation) before training
"""

import gc

import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_sparse import SparseTensor

from evaluation import set_graph_for_modularity

OGB_DATASETS = ["ogbn-arxiv", "ogbn-products", "ogbn-papers100m"]
AMAZON_DATASETS = ["photo", "computers"]
PLANETOID_DATASETS = ["cora", "citeseer"]


def load_data(dataset_name, save_path):
    """Load a dataset and return (data, num_classes).

    `data` holds `x`, `edge_index` and `y`; `y` is always 1-D.
    """
    if dataset_name in OGB_DATASETS:
        dataset = PygNodePropPredDataset(
            name=dataset_name, root=f"{save_path}/data/ogb"
        )
        data = dataset[0]
        data.y = data.y.squeeze(-1)
        scaler = StandardScaler()
        data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
        num_classes = dataset.num_classes

    elif dataset_name == "mag_scholar_c":
        print("正在加载 MAG-Scholar-C 并进行 SVD 降维...")
        data_dir = f"{save_path}/data/mag_scholar_c"

        feat_sp = sp.load_npz(f"{data_dir}/features.npz").tocsr()
        svd = TruncatedSVD(n_components=128, random_state=42)
        x_dense = svd.fit_transform(feat_sp)
        del feat_sp
        gc.collect()

        x = torch.tensor(x_dense, dtype=torch.float)
        x = F.normalize(x, p=1, dim=1)
        del x_dense
        gc.collect()

        edge_index = torch.load(f"{data_dir}/edge_index.pt").to(torch.int32)
        y = torch.load(f"{data_dir}/labels.pt")
        data = Data(x=x, edge_index=edge_index, y=y)
        num_classes = 8

    elif dataset_name in AMAZON_DATASETS:
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Amazon(
            root=f"{save_path}/data/{dataset_name}",
            name=dataset_name.capitalize(),
            transform=transform,
        )
        data = dataset[0]
        num_classes = dataset.num_classes

    elif dataset_name in PLANETOID_DATASETS:
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Planetoid(
            root=f"{save_path}/data/{dataset_name}",
            name=dataset_name.capitalize(),
            transform=transform,
        )
        data = dataset[0]
        num_classes = dataset.num_classes

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return data, num_classes


def preprocess_smooth_features_large(x, edge_index, k_steps, alpha, device="cpu"):
    """Propagate features with the symmetrically normalized adjacency matrix.

    Each step computes h <- alpha * h + (1 - alpha) * A_norm @ h.
    """
    if k_steps == 0:
        return x

    N = x.shape[0]
    edge_index_dev = edge_index.to(device).long()
    x = x.to(device)

    adj_t = SparseTensor(
        row=edge_index_dev[0],
        col=edge_index_dev[1],
        sparse_sizes=(N, N),
        is_sorted=False,
    )
    deg = adj_t.sum(dim=1).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t.mul(deg_inv_sqrt.view(1, -1)).mul(deg_inv_sqrt.view(-1, 1))
    del deg, deg_inv_sqrt

    h0 = x
    h_new = torch.empty_like(x)

    for step in range(k_steps):
        h_next = adj_t @ h0
        h_new.copy_(h0).mul_(alpha).add_(h_next, alpha=(1 - alpha))
        del h_next
        h0, h_new = h_new, h0

    result = h0
    del adj_t, h_new, edge_index_dev
    gc.collect()
    return result


def prepare_data(dataset_name, save_path, k_steps, alpha):
    """Full data preparation used by training.

    Returns (data, num_classes, y_true, num_nodes, global_deg, global_m2).
    Smoothed features are stored in `data.x_input`; the raw `data.x` is freed.
    """
    data, num_classes = load_data(dataset_name, save_path)

    num_nodes = data.x.shape[0]
    y_true = data.y.numpy()

    if data.edge_index.dtype != torch.int32:
        data.edge_index = data.edge_index.to(torch.int32)

    set_graph_for_modularity(data.edge_index, num_nodes)

    x_smooth = preprocess_smooth_features_large(
        data.x, data.edge_index, k_steps=k_steps, alpha=alpha
    )
    data.x_input = x_smooth
    del data.x
    del x_smooth
    gc.collect()

    global_deg = torch.bincount(data.edge_index[0].long(), minlength=num_nodes).float()
    global_m2 = global_deg.sum().clamp(min=1.0)

    return data, num_classes, y_true, num_nodes, global_deg, global_m2
#（注：内容由AI生成）

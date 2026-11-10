"""Hyperbolic geometry, prototype encoder and the losses defined on it.

Contains:
  * `SafePoincareSpace` - numerically guarded Poincare ball ops (exp/log maps,
    distances), plus a shared instance `_manifold`
  * `compute_membership_profile` - confidence gate from prototype distances
  * `HyperbolicPrototypeEncoder` - MLP trunk mapped into the Poincare ball
  * losses: reconstruction (edge distance + negative sampling), hyperbolic
    modularity (full graph), scalable hyperbolic spectral clustering (batches),
    and the cluster-balance regularizer
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans


class SafePoincareSpace:
    """Poincare ball with curvature c and norm clipping."""

    def __init__(self, c=1.0, eps=1e-5, max_norm=0.98):
        self.c = c
        self.eps = eps
        self.max_norm = max_norm

    def exp_map_zero(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = np.sqrt(self.c)
        r = torch.tanh(sqrt_c * norm_x) / sqrt_c
        r = r.clamp(max=self.max_norm)
        return r * (x / norm_x)

    def log_map_zero(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = np.sqrt(self.c)
        safe_input = torch.clamp(norm_x * sqrt_c, max=1.0 - 1e-6)
        coeff = torch.atanh(safe_input) / (sqrt_c * norm_x + 1e-8)
        return coeff * x

    def poincare_dist(self, x, y):
        x_sq = torch.sum(x**2, dim=-1, keepdim=True).clamp(max=self.max_norm**2)
        y_sq = torch.sum(y**2, dim=-1, keepdim=True).clamp(max=self.max_norm**2)
        sqdist = torch.cdist(x, y, p=2) ** 2
        num = 2 * self.c * sqdist
        den = (1.0 - self.c * x_sq) * (1.0 - self.c * y_sq.transpose(-2, -1))
        arg = 1.0 + num / den.clamp(min=1e-5)
        arg = arg.clamp(min=1.0 + 1e-5, max=1e4)
        return (1.0 / np.sqrt(self.c)) * torch.acosh(arg)

    def poincare_dist_edges(self, u, v):
        u_sq = torch.sum(u**2, dim=-1, keepdim=True).clamp(max=self.max_norm**2)
        v_sq = torch.sum(v**2, dim=-1, keepdim=True).clamp(max=self.max_norm**2)
        sqdist = torch.sum((u - v) ** 2, dim=-1, keepdim=True)
        num = 2 * self.c * sqdist
        den = (1.0 - self.c * u_sq) * (1.0 - self.c * v_sq)
        arg = 1.0 + num / den.clamp(min=1e-5)
        arg = arg.clamp(min=1.0 + 1e-5, max=1e4)
        return (1.0 / np.sqrt(self.c)) * torch.acosh(arg).squeeze(-1)


_manifold = SafePoincareSpace()


def compute_membership_profile(dists, gate_alpha=0.9):
    """Confidence of the closest prototype from the two smallest distances."""
    K = dists.size(1)
    if K >= 2:
        sorted_d, _ = torch.sort(dists, dim=1)
        d1 = sorted_d[:, 0:1]
        d2 = sorted_d[:, 1:2]
    else:
        d1 = dists
        d2 = dists

    d1 = d1.clamp(min=1e-8)
    d2 = d2.clamp(min=1e-8)

    s = (d2 - d1) / (d2 + d1 + 1e-8)
    exp_gate = torch.exp(-d1.pow(2) / (0.5**2))
    gate = gate_alpha * exp_gate + (1.0 - gate_alpha) * s
    return s, exp_gate, gate


class HyperbolicPrototypeEncoder(nn.Module):
    """MLP encoder with learnable (or EMA) prototypes living on the ball."""

    def __init__(
        self,
        in_features,
        hidden_dim,
        num_clusters,
        middle_dim=512,
        drop_out=0.1,
        use_ema_prototype=False,
        ema_momentum=0.99,
        is_large_graph=False,
        seed=1,
    ):
        super().__init__()
        self.manifold = _manifold
        self.num_clusters = num_clusters
        self.middle_dim = middle_dim
        self.drop_out = drop_out
        self.use_ema_prototype = use_ema_prototype
        self.ema_momentum = ema_momentum
        self.is_large_graph = is_large_graph
        self.seed = seed

        self.encoder = nn.Sequential(
            nn.Linear(in_features, middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(middle_dim, hidden_dim),
        )

        if use_ema_prototype:
            self.register_buffer(
                "prototype_tangent", torch.randn(num_clusters, hidden_dim)
            )
            self.register_buffer("dq_sum_ema", torch.zeros(num_clusters))
            self.register_buffer(
                "prototype_initialized", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.tangent_centers = nn.Parameter(torch.randn(num_clusters, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h_tangent = self.encoder(x)
        norm = h_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = h_tangent / norm
        radius = 0.3 + 0.65 * torch.tanh(norm)
        return self.manifold.exp_map_zero(direction * radius)

    def get_centroids(self):
        if self.use_ema_prototype:
            c = self.prototype_tangent
        else:
            c = self.tangent_centers
        c_norm = F.normalize(c, p=2, dim=-1)
        return self.manifold.exp_map_zero(c_norm)

    def compute_q(self, z):
        centroids = self.get_centroids()
        dists = self.manifold.poincare_dist(z, centroids)
        q = 1.0 / (1.0 + dists)
        q = q ** ((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q, dists

    @torch.no_grad()
    def update_prototypes_from_batch(self, z, q):
        if not self.use_ema_prototype:
            return
        z_tangent = self.manifold.log_map_zero(z)
        w = q.sum(dim=0)
        w_safe = w.clamp(min=1e-6)
        local_proto = (q.t() @ z_tangent) / w_safe.unsqueeze(1)
        mask = (w > 1e-6).unsqueeze(1)
        old = self.prototype_tangent.clone()
        new = self.ema_momentum * old + (1 - self.ema_momentum) * local_proto
        self.prototype_tangent.data = torch.where(mask, new, old)

    @torch.no_grad()
    def init_centroids_with_kmeans(self, z_numpy):
        z_tangent = self.manifold.log_map_zero(torch.from_numpy(z_numpy)).numpy()

        if self.is_large_graph:
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                batch_size=8192,
                n_init=10,
                max_iter=100,
                random_state=self.seed,
            )
        else:
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                n_init=20,
                random_state=self.seed,
            )
        kmeans.fit(z_tangent)

        del z_tangent
        gc.collect()

        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        if self.use_ema_prototype:
            self.prototype_tangent.data = centers.to(self.prototype_tangent.device)
            self.dq_sum_ema.zero_()
            self.prototype_initialized.fill_(1)
        else:
            self.tangent_centers.data = centers.to(self.tangent_centers.device)


def compute_graph_rec_loss(
    z,
    edge_index,
    num_nodes,
    num_neg=50000,
    hard_ratio=0.3,
    margin_ratio=0.5,
    edge_chunk=500000,
):
    """Edge reconstruction loss in hyperbolic space with degree-based negatives."""
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=z.device)

    E = edge_index.size(1)
    pos_loss_sum = z.new_zeros(())
    for i in range(0, E, edge_chunk):
        end = min(i + edge_chunk, E)
        ei = edge_index[:, i:end]
        pos_d = _manifold.poincare_dist_edges(z[ei[0]], z[ei[1]])
        pos_loss_sum = pos_loss_sum + pos_d.sum()
    pos_mean = pos_loss_sum / E

    num_neg = min(num_neg, num_nodes * (num_nodes - 1) // 2)

    deg = (
        torch.bincount(edge_index[0].long(), minlength=num_nodes)
        + torch.bincount(edge_index[1].long(), minlength=num_nodes)
    ).float()
    deg = deg.clamp(min=1)
    p = deg.pow(0.75)
    p = p / p.sum()

    rand_u = torch.multinomial(p, num_neg, replacement=True)
    rand_v = torch.multinomial(p, num_neg, replacement=True)

    mask_self = rand_u != rand_v
    rand_u = rand_u[mask_self]
    rand_v = rand_v[mask_self]

    if hard_ratio > 0:
        cand_u = z[rand_u]
        cand_v = z[rand_v]
        cand_dist = _manifold.poincare_dist_edges(cand_u, cand_v)
        hard_num = int(rand_u.size(0) * hard_ratio)
        _, top_idx = torch.topk(cand_dist, hard_num, largest=False)
        mask = torch.ones(rand_u.size(0), dtype=torch.bool, device=z.device)
        mask[top_idx] = False
        remain_idx = mask.nonzero(as_tuple=True)[0]
        neg_u = torch.cat([rand_u[top_idx], rand_u[remain_idx]])
        neg_v = torch.cat([rand_v[top_idx], rand_v[remain_idx]])
    else:
        neg_u = rand_u
        neg_v = rand_v

    neg_dist = _manifold.poincare_dist_edges(z[neg_u], z[neg_v])

    with torch.no_grad():
        pos_mean_val = pos_mean.detach()
        neg_mean_val = neg_dist.mean().detach()
        margin = pos_mean_val + margin_ratio * (neg_mean_val - pos_mean_val)
        margin = torch.clamp(margin, min=0.3, max=10.0)

    neg_loss_sum = z.new_zeros(())
    for i in range(0, neg_u.size(0), edge_chunk):
        end = min(i + edge_chunk, neg_u.size(0))
        nd = _manifold.poincare_dist_edges(z[neg_u[i:end]], z[neg_v[i:end]])
        neg_loss_sum = neg_loss_sum + F.relu(margin - nd).sum()
    neg_mean_loss = neg_loss_sum / neg_u.size(0)

    return pos_mean + neg_mean_loss


def hyperbolic_modularity_loss(
    z,
    q,
    edge_index,
    num_nodes,
    centroids,
    global_deg=None,
    global_m2=None,
    gate_alpha=0.9,
):
    """Modularity of the soft assignment, weighted by hyperbolic edge proximity."""
    K = q.size(1)
    edge_dist = _manifold.poincare_dist_edges(z[edge_index[0]], z[edge_index[1]])
    weight = torch.exp(-edge_dist.pow(2))

    dists = _manifold.poincare_dist(z, centroids)
    _, _, gate = compute_membership_profile(dists, gate_alpha=gate_alpha)

    uniform = torch.full_like(q, 1.0 / K)
    q_tilde = gate * q + (1 - gate) * uniform

    if global_deg is not None and global_m2 is not None:
        deg = global_deg
        m2 = global_m2
    else:
        deg = torch.bincount(edge_index[0].long(), minlength=num_nodes).float()
        m2 = deg.sum().clamp(min=1.0)

    edge_q_dot = (q_tilde[edge_index[0]] * q_tilde[edge_index[1]]).sum(dim=1)
    term1 = (weight * edge_q_dot).sum() / m2

    dq_sum = (deg.unsqueeze(1) * q_tilde).sum(dim=0)
    term2 = (dq_sum**2).sum() / (m2**2)

    return -(term1 - term2)


def scalable_hsc_loss(
    z,
    q,
    dists,
    edge_index,
    global_deg_batch,
    gate_alpha=0.9,
    edge_chunk=500000,
):
    """Batched version of the hyperbolic modularity loss on induced subgraphs."""
    K = q.size(1)
    E = edge_index.size(1)

    _, _, gate = compute_membership_profile(dists, gate_alpha=gate_alpha)
    uniform = torch.full_like(q, 1.0 / K)
    q_tilde = gate * q + (1 - gate) * uniform
    batch_m2 = global_deg_batch.sum().clamp(min=1.0)

    term1_sum = z.new_zeros(())
    for i in range(0, E, edge_chunk):
        end = min(i + edge_chunk, E)
        ei = edge_index[:, i:end]
        edge_dist = _manifold.poincare_dist_edges(z[ei[0]], z[ei[1]])
        weight = torch.exp(-edge_dist.pow(2))
        edge_q_dot = (q_tilde[ei[0]] * q_tilde[ei[1]]).sum(dim=1)
        term1_sum = term1_sum + (weight * edge_q_dot).sum()
    term1 = term1_sum / batch_m2

    dq = (global_deg_batch.unsqueeze(1) * q_tilde).sum(dim=0)
    term2 = (dq / batch_m2).pow(2).sum()

    return -(term1 - term2)


def balance_loss(q, eps=1e-8):
    """KL divergence between the mean assignment distribution and the uniform prior."""
    marginal = q.mean(dim=0).clamp(min=eps)
    uniform = torch.full_like(marginal, 1.0 / marginal.numel())
    return F.kl_div(marginal.log(), uniform, reduction="sum")
#（注：内容由AI生成）

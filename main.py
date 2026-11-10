"""Training entry point.

The training pipeline only lives here; everything else is split into modules:

    data_preprocess.py  - loading and feature smoothing
    graph_sampling.py   - random-walk subgraph sampling
    hyperbolic.py       - Poincare ball geometry, encoder, losses
    evaluation.py       - modularity and clustering metrics
    utils.py            - seeding and checkpoint helpers
"""

import gc
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data_preprocess import prepare_data
from graph_sampling import (
    GraphStorage,
    build_batches,
    weighted_random_walk_partition,
)
from hyperbolic import (
    HyperbolicPrototypeEncoder,
    balance_loss,
    compute_graph_rec_loss,
    hyperbolic_modularity_loss,
    scalable_hsc_loss,
)
from utils import load_checkpoint, save_checkpoint, set_seed

# ==========================================
# Configuration
# ==========================================
save_path = "/root/autodl-tmp"
SAVE_DIR = "./checkpoints"
seed = 1

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_name = "ogbn-products".lower()
is_large_graph = True
use_ema_prototype = is_large_graph

lambda_bal = 0.05
lambda_rec = 0.05
k_steps = 2
alpha = 0.5
ema_momentum = None

if dataset_name == "ogbn-arxiv":
    WALK_ROOTS_PER_BATCH = 8192
    WALK_NUM_BATCHES = 50
    WALK_T_PER_ROOT = 20
    WALK_DEPTH = 5
    WALK_TAU_FEAT = 0.9
    WALK_MAX_BATCH = 100000
    WALK_MIN_BATCH = 1000
    ema_momentum = 0.99
    epochs = 180
    warmup_epochs = 30
    lr_warmup = 1e-3
    wd_warmup = 1e-3
    lr_main = 5e-4
    wd_main = 1e-4
    eta_min_ratio = 0.01
    hidden_dim = 128
    middle_dim = 512
    drop_out = 0.1

elif dataset_name == "ogbn-products":
    WALK_ROOTS_PER_BATCH = 4096
    WALK_NUM_BATCHES = 50
    WALK_T_PER_ROOT = 30
    WALK_DEPTH = 4
    WALK_TAU_FEAT = 0.9
    WALK_MAX_BATCH = 100000
    WALK_MIN_BATCH = 1000
    ema_momentum = 0.995
    epochs = 180
    warmup_epochs = 80
    lr_warmup = 1e-2
    wd_warmup = 0
    lr_main = 5e-4
    wd_main = 0
    eta_min_ratio = 1.0
    hidden_dim = 128
    middle_dim = 512
    drop_out = 0.1

elif dataset_name == "citeseer":
    epochs = 400
    warmup_epochs = 150
    lr_warmup = 1e-4
    wd_warmup = 1e-2
    lr_main = 5e-5
    wd_main = 1e-2
    eta_min_ratio = 0.1
    hidden_dim = 256
    middle_dim = 512
    drop_out = 0.5

elif dataset_name == "cora":
    epochs = 180
    warmup_epochs = 80
    lr_warmup = 5e-4
    wd_warmup = 1e-3
    lr_main = 5e-5
    wd_main = 1e-4
    eta_min_ratio = 0.01
    hidden_dim = 64
    middle_dim = 512
    drop_out = 0.1

elif dataset_name == "photo":
    epochs = 350
    warmup_epochs = 100
    lr_warmup = 3e-3
    wd_warmup = 1e-2
    lr_main = 4e-5
    wd_main = 1e-2
    eta_min_ratio = 0.1
    hidden_dim = 256
    middle_dim = 512
    drop_out = 0.0

elif dataset_name == "computers":
    epochs = 500
    warmup_epochs = 150
    lr_warmup = 5e-3
    wd_warmup = 5e-3
    lr_main = 4e-5
    wd_main = 1e-2
    eta_min_ratio = 0.01
    hidden_dim = 256
    middle_dim = 512
    drop_out = 0.1

elif dataset_name == "mag_scholar_c":
    WALK_ROOTS_PER_BATCH = 4096
    WALK_NUM_BATCHES = 50
    WALK_T_PER_ROOT = 20
    WALK_DEPTH = 4
    WALK_TAU_FEAT = 0.9
    WALK_MAX_BATCH = 100000
    WALK_MIN_BATCH = 1000
    ema_momentum = 0.995
    epochs = 200
    warmup_epochs = 50
    lr_warmup = 1e-2
    wd_warmup = 0
    lr_main = 5e-4
    wd_main = 0
    eta_min_ratio = 1.0
    hidden_dim = 128
    middle_dim = 512
    drop_out = 0.1

set_seed(seed)


def train_adaptive_pipeline(
    dataset_name,
    hidden_dim,
    epochs,
    warmup_epochs,
    is_large_graph=False,
    load_pretrained=True,
    force_retrain=False,
):
    print(f"--> 设备: {device} | 模式: {'采样 (MMRW)' if is_large_graph else '全图'}")

    # ---------- Data loading and preprocessing ----------
    data, num_classes, y_true, num_nodes, global_deg, global_m2 = prepare_data(
        dataset_name=dataset_name,
        save_path=save_path,
        k_steps=k_steps,
        alpha=alpha,
    )

    # ---------- Model ----------
    model = HyperbolicPrototypeEncoder(
        in_features=data.x_input.shape[1],
        hidden_dim=hidden_dim,
        num_clusters=num_classes,
        middle_dim=middle_dim,
        drop_out=drop_out,
        use_ema_prototype=use_ema_prototype,
        ema_momentum=ema_momentum,
        is_large_graph=is_large_graph,
        seed=seed,
    ).to(device)

    ckpt_path = os.path.join(SAVE_DIR, f"{dataset_name}.pt")

    if not force_retrain and load_pretrained and os.path.exists(ckpt_path):
        print(f"--> 发现已保存的模型: {ckpt_path}，直接加载退出。")
        load_checkpoint(ckpt_path, model, device)
        return model, data, y_true

    # ---------- Optimizers ----------
    optimizer_warmup = torch.optim.Adam(
        model.parameters(), lr=lr_warmup, weight_decay=wd_warmup
    )
    optimizer_main = torch.optim.AdamW(
        model.parameters(), lr=lr_main, weight_decay=wd_main
    )
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=epochs - warmup_epochs, eta_min=lr_main * eta_min_ratio
    )

    # ---------- MMRW preprocessing for large graphs ----------
    batch_data = None
    if is_large_graph:
        print("\n--> 构建 GraphStorage 并执行 MMRW 采样...")
        graph_storage = GraphStorage(data.edge_index, num_nodes, data.x_input, device)
        batches = weighted_random_walk_partition(
            graph_storage,
            roots_per_batch=WALK_ROOTS_PER_BATCH,
            num_batches=WALK_NUM_BATCHES,
            t_per_root=WALK_T_PER_ROOT,
            walk_depth=WALK_DEPTH,
            tau_feat=WALK_TAU_FEAT,
            max_batch=WALK_MAX_BATCH,
            min_batch=WALK_MIN_BATCH,
            seed=seed,
        )
        batch_data = build_batches(graph_storage, batches, data.x_input)
        del batches
        graph_storage.release()
        del graph_storage
        gc.collect()

    # ==========================================
    # Training loop
    # ==========================================
    max_norm = 2.0
    pbar = tqdm(range(1, epochs + 1), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        total_loss = 0.0
        is_warmup = epoch <= warmup_epochs
        optimizer = optimizer_warmup if is_warmup else optimizer_main

        if not is_large_graph:
            optimizer.zero_grad()
            edge_index = data.edge_index.to(device).long()
            z = model(data.x_input.to(device))
            loss_rec = compute_graph_rec_loss(
                z,
                edge_index,
                num_nodes,
                num_neg=num_nodes,
                hard_ratio=0.3,
                margin_ratio=0.7,
            )

            if is_warmup:
                loss = loss_rec
            else:
                q, _ = model.compute_q(z)
                centroids = model.get_centroids()
                loss_mod = hyperbolic_modularity_loss(
                    z,
                    q,
                    edge_index,
                    num_nodes,
                    centroids,
                    global_deg.to(device),
                    global_m2.to(device),
                )
                loss_bal = balance_loss(q)
                loss = loss_mod + lambda_bal * loss_bal + lambda_rec * loss_rec

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            if not is_warmup:
                model.update_prototypes_from_batch(z.detach(), q.detach())
                scheduler_main.step()
            epoch_loss = loss.item()

        else:
            epoch_proto_accum = None
            epoch_proto_weight = None
            idx = np.random.permutation(len(batch_data))
            for i in idx:
                b = batch_data[i]
                x_b = b["x"].to(device)
                ei_b = b["edge_index"].to(device).long()
                n_b = b["num_nodes"]
                ids_b = b["node_ids"]

                optimizer.zero_grad()
                z = model(x_b)
                loss_rec = compute_graph_rec_loss(
                    z,
                    ei_b,
                    n_b,
                    num_neg=min(500000, n_b * 5),
                    hard_ratio=0.3,
                    margin_ratio=0.5,
                )

                if is_warmup:
                    loss = loss_rec
                else:
                    q, dists = model.compute_q(z)
                    gd_batch = global_deg[ids_b].to(device)
                    loss_mod = scalable_hsc_loss(z, q, dists, ei_b, gd_batch)
                    loss_bal = balance_loss(q)
                    loss = loss_mod + lambda_bal * loss_bal + lambda_rec * loss_rec

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

                if not is_warmup:
                    with torch.no_grad():
                        z_tan = model.manifold.log_map_zero(z.detach())
                        w = q.detach().sum(dim=0)
                        w_safe = w.clamp(min=1e-6)
                        local_proto = (q.detach().t() @ z_tan) / w_safe.unsqueeze(1)

                        if epoch_proto_accum is None:
                            epoch_proto_accum = local_proto * w.unsqueeze(1)
                            epoch_proto_weight = w.clone()
                        else:
                            epoch_proto_accum = (
                                epoch_proto_accum + local_proto * w.unsqueeze(1)
                            )
                            epoch_proto_weight = epoch_proto_weight + w

                total_loss += loss.item()
                del x_b, ei_b, z, loss_rec, loss
                if not is_warmup:
                    del q, dists, gd_batch, loss_mod, loss_bal

            if not is_warmup and epoch_proto_accum is not None:
                with torch.no_grad():
                    epoch_proto_weight_safe = epoch_proto_weight.clamp(min=1e-6)
                    avg_proto = epoch_proto_accum / epoch_proto_weight_safe.unsqueeze(1)
                    mask = (epoch_proto_weight > 1e-6).unsqueeze(1)
                    old = model.prototype_tangent.clone()
                    new = (
                        model.ema_momentum * old + (1 - model.ema_momentum) * avg_proto
                    )
                    model.prototype_tangent.data = torch.where(mask, new, old)
                del epoch_proto_accum, epoch_proto_weight

            if not is_warmup:
                scheduler_main.step()
            epoch_loss = total_loss / len(batch_data)

        # End of warmup: initialize prototypes
        if epoch == warmup_epochs:
            print("\n--> Warmup 结束，开始初始化原型 ...")
            selected_batches = (
                random.sample(batch_data, min(20, len(batch_data)))
                if is_large_graph
                else None
            )

            z_list = []
            with torch.no_grad():
                if is_large_graph:
                    for b in selected_batches:
                        x_b = b["x"].to(device)
                        z_b = model(x_b)
                        z_list.append(z_b.cpu())
                else:
                    z_b = model(data.x_input.to(device))
                    z_list.append(z_b.cpu())

            z_sample = torch.cat(z_list, dim=0)
            del z_list
            gc.collect()

            model.init_centroids_with_kmeans(z_sample.numpy())
            del z_sample
            gc.collect()
            print("--> 原型初始化完成。")

        pbar.set_postfix(
            {"Loss": f"{epoch_loss:.4f}", "Phase": "warmup" if is_warmup else "main"}
        )

    save_path_final = os.path.join(f"{SAVE_DIR}/{dataset_name}", "final.pt")
    save_checkpoint(save_path_final, model, extra={"dataset_name": dataset_name})
    print(f"\n--> 训练完成！最终模型已保存至: {save_path_final}")

    return model, data, y_true


if __name__ == "__main__":
    model, data, y_true = train_adaptive_pipeline(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        is_large_graph=is_large_graph,
        load_pretrained=False,
        force_retrain=True,
    )

# Reliability-Aware Hyperbolic Community Detection for Large-Scale Attributed Graphs

## 1. Layout

| File | Responsibility |
| --- | --- |
| `main.py` | Configuration block + `train_adaptive_pipeline` (warmup / main training / prototype init / checkpointing) |
| `data_preprocess.py` | Dataset loading (OGB, MAG-Scholar-C, Amazon, Planetoid), label and edge-index normalisation, degree statistics, feature smoothing |
| `graph_sampling.py` | `GraphStorage` (CSR whole-graph storage + induced-subgraph extraction), `weighted_random_walk_partition` (MMRW sampling), `build_batches` |
| `hyperbolic.py` | `SafePoincareSpace` (exp/log maps and distances on the Poincare ball), membership gate, `HyperbolicPrototypeEncoder`, the three losses plus the balance regulariser |
| `evaluation.py` | Global modularity statistics, `modularity_score`, ARI / NMI / ACC, hard assignments, `evaluate_model`, with a CLI entry point |
| `utils.py` | `set_seed`, `save_checkpoint`, `load_checkpoint` |
| `resource_monitor.py` | Training resource monitor (CPU / memory / GPU utilisation, VRAM, power, temperature); optional |
| `requirements.txt` | Dependencies |

## 2. Installation

```bash
pip install -r requirements.txt
```

Pinned versions that matter: `torch==2.2.2+cu118`, `torch_geometric==2.4.0`, `torch_sparse==0.6.18+pt22cu118`, `ogb==1.3.6`, `scikit-learn==1.7.2`, `numba==0.67.0`.

`torch_sparse` and `torch_scatter` must match the torch / CUDA build. If plain pip fails, use the PyG wheel index:

```bash
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
```

`numba` is optional. Without it, the cumulative-weight table used by the random walks falls back to a pure-Python loop — identical results, only slower.

## 3. Data

| Dataset | Location | Notes |
| --- | --- | --- |
| `ogbn-arxiv` / `ogbn-products` | `{save_path}/data/ogb` | Downloaded automatically on first run |
| `cora` / `citeseer` | `{save_path}/data/{name}` | Downloaded automatically |
| `photo` / `computers` | `{save_path}/data/{name}` | Downloaded automatically |
| `mag_scholar_c` | `{save_path}/data/mag_scholar_c` | Requires `features.npz` (sparse features), `edge_index.pt`, `labels.pt` placed manually |

`save_path` is configured at the top of `main.py` (default `/root/autodl-tmp`). MAG-Scholar-C features are reduced to 128 dimensions with TruncatedSVD and then L1-normalised.

## 4. Running

```bash
python main.py
```

To switch datasets, edit the two lines at the top of `main.py`:

```python
dataset_name = "ogbn-products".lower()
is_large_graph = True          # True: MMRW-sampled subgraphs; set False for cora / citeseer / photo / computers
```

Outputs:

- `{SAVE_DIR}/{dataset_name}.pt` — cache used to skip training when a checkpoint already exists
- `{SAVE_DIR}/{dataset_name}/final.pt` — final model written at the end of training

`SAVE_DIR` defaults to `./checkpoints`.

## 5. Configuration (top of `main.py`)

### 5.1 Runtime parameters

| Variable | Default | Meaning |
| --- | --- | --- |
| `save_path` | `/root/autodl-tmp` | Dataset root directory |
| `SAVE_DIR` | `./checkpoints` | Checkpoint output directory |
| `seed` | `1` | Global seed (Python / NumPy / Torch); also seeds the walk RNG and KMeans |
| `dataset_name` | `ogbn-products` | Selects the hyper-parameter branch below |
| `is_large_graph` | `True` | `True`: train on sampled subgraphs; `False`: train on the full graph |
| `use_ema_prototype` | `= is_large_graph` | `True`: prototypes updated by EMA (buffers); `False`: prototypes are learnable parameters |
| `lambda_bal` | `0.05` | Weight of the cluster-balance KL term |
| `lambda_rec` | `0.05` | Weight of the graph reconstruction loss |
| `k_steps` | `2` | Number of feature-smoothing steps (`0` disables smoothing) |
| `alpha` | `0.5` | Residual coefficient: `h <- alpha * h + (1 - alpha) * A_norm @ h` |
| `ema_momentum` | set inside the branch | Prototype EMA momentum (`0.995` for `ogbn-products`) |

### 5.2 Per-dataset hyper-parameters

| Variable | Meaning |
| --- | --- |
| `epochs` | Total number of epochs |
| `warmup_epochs` | Epochs trained with the reconstruction loss only; prototypes are KMeans-initialised when warmup ends |
| `lr_warmup` / `wd_warmup` | Adam learning rate / weight decay during warmup |
| `lr_main` / `wd_main` | AdamW learning rate / weight decay in the main phase |
| `eta_min_ratio` | Cosine-annealing floor = `lr_main * eta_min_ratio` |
| `hidden_dim` | Encoder output dimension (i.e. the tangent-space dimension) |
| `middle_dim` | Encoder hidden width |
| `drop_out` | Encoder dropout probability |
| `WALK_ROOTS_PER_BATCH` | Number of random-walk roots per batch |
| `WALK_NUM_BATCHES` | Number of batches |
| `WALK_T_PER_ROOT` | Walks started from each root |
| `WALK_DEPTH` | Length of a single walk |
| `WALK_TAU_FEAT` | Feature-similarity temperature: `exp(cos / tau)` |
| `WALK_MAX_BATCH` / `WALK_MIN_BATCH` | Upper / lower bound on batch size (batches below the lower bound are dropped) |

Current values:

| Dataset | epochs | warmup | lr_warmup | lr_main | hidden | drop_out |
| --- | --- | --- | --- | --- | --- | --- |
| `ogbn-arxiv` | 180 | 30 | 1e-3 | 5e-4 | 128 | 0.1 |
| `ogbn-products` | 180 | 80 | 1e-2 | 5e-4 | 128 | 0.1 |
| `citeseer` | 400 | 150 | 1e-4 | 5e-5 | 256 | 0.5 |
| `cora` | 180 | 80 | 5e-4 | 5e-5 | 64 | 0.1 |
| `photo` | 350 | 100 | 3e-3 | 4e-5 | 256 | 0.0 |
| `computers` | 500 | 150 | 5e-3 | 4e-5 | 256 | 0.1 |
| `mag_scholar_c` | 200 | 50 | 1e-2 | 5e-4 | 128 | 0.1 |

To add a dataset, append one more branch to the chain, or add a loader branch in `data_preprocess.load_data`.

> Note: `WALK_*` is only defined in the `ogbn-arxiv`, `ogbn-products` and `mag_scholar_c` branches. Setting `is_large_graph = True` on a small-graph dataset raises `NameError` because those names do not exist; setting it to `False` on a large dataset is harmless. This is pre-existing behaviour and was deliberately left untouched.

### 5.3 Checkpoint behaviour

```python
model, data, y_true = train_adaptive_pipeline(
    dataset_name=dataset_name, hidden_dim=hidden_dim,
    epochs=epochs, warmup_epochs=warmup_epochs,
    is_large_graph=is_large_graph,
    load_pretrained=False,   # True + existing cache: load and return without training
    force_retrain=True,      # True: always retrain from scratch
)
```

When `load_pretrained=True` and `force_retrain=False` and `{SAVE_DIR}/{dataset_name}.pt` exists, training is skipped and the cached model is returned.

## 6. Evaluation

`evaluation.py` also works as a standalone script. It infers `in_features / middle_dim / hidden_dim / cluster count / whether prototypes are EMA-based` from the checkpoint, so no hyper-parameters have to be filled in by hand:

```bash
# Full-graph dataset
python evaluation.py --dataset-name cora \
    --checkpoint ./checkpoints/cora/final.pt \
    --data-root /root/autodl-tmp

# Large graph (inference follows the sampled batches, modularity uses the full graph)
python evaluation.py --dataset-name ogbn-products --is-large-graph \
    --checkpoint ./checkpoints/ogbn-products/final.pt
```

- `ARI` / `NMI`: agreement with the ground-truth labels; `ACC`: accuracy after Hungarian matching.
- `Modularity`: Newman modularity computed from the cached global degree statistics and `2m`, consistent with `global_deg / global_m2` used inside the training losses.
- In large-graph mode nodes are encoded batch by batch; nodes that never fall into a batch stay at `-1` and are excluded from the metrics.
- Sampling arguments default to the `ogbn-products` values and can be overridden with `--roots-per-batch / --num-batches / --t-per-root / --walk-depth / --tau-feat / --max-batch / --min-batch`.
- Other flags: `--k-steps`, `--alpha` (must match training), `--drop-out`, `--ema-momentum`, `--device`, `--data-root`, `--save-dir`.

Evaluation is a standalone utility. **The training pipeline does not call it**, so the original training logic is untouched.

## 7. Resource monitor (optional)

```python
from resource_monitor import ResourceMonitor

monitor = ResourceMonitor(interval=1.0)
monitor.start()

# ... training ...

monitor.stop()
monitor.report()               # human-readable summary
summary = monitor.summary()    # structured dict, easy to dump to disk
```

It tracks CPU utilisation, system memory, per-GPU utilisation / VRAM / power / temperature, and PyTorch peak memory.

## 8. Module API cheat sheet

```python
# Data
from data_preprocess import prepare_data
data, num_classes, y_true, num_nodes, global_deg, global_m2 = prepare_data(
    dataset_name, save_path, k_steps, alpha)
# smoothed features are stored in data.x_input

# Sampling
from graph_sampling import GraphStorage, weighted_random_walk_partition, build_batches
graph_storage = GraphStorage(data.edge_index, num_nodes, data.x_input, device)
batches = weighted_random_walk_partition(graph_storage, roots_per_batch=..., seed=seed)
batch_data = build_batches(graph_storage, batches, data.x_input)   # [{node_ids, x, edge_index, num_nodes}]

# Model and losses
from hyperbolic import HyperbolicPrototypeEncoder, compute_graph_rec_loss, \
     hyperbolic_modularity_loss, scalable_hsc_loss, balance_loss
model = HyperbolicPrototypeEncoder(in_features, hidden_dim, num_clusters,
        middle_dim=middle_dim, drop_out=drop_out,
        use_ema_prototype=..., ema_momentum=..., is_large_graph=..., seed=seed)

# Evaluation
from evaluation import evaluate_model
report = evaluate_model(model, y_true, num_classes=num_classes,
        x_input=data.x_input, edge_index=data.edge_index,
        num_nodes=num_nodes, device=device)
```

## 9. Notes on this split

- Every moved function and class keeps its body identical to the original; only English docstrings were added.
- Places where the encoder previously read the module-level globals `middle_dim`, `drop_out`, `is_large_graph` and `seed` now take them as constructor arguments. The values still come from the configuration block at the top of `main.py`, and the computation is unchanged.
- Memory behaviour is preserved: `data.x` is deleted after smoothing, the intermediate tensors discarded in the original are still discarded, and every `gc.collect()` call sits where it did before.
- All Chinese comments were removed and rewritten as English comments / docstrings. Console messages printed at runtime were kept in their original wording so that output stays identical to the original run; translating them does not affect the logic.
- New in this split: `evaluation.py` turns the previously imported-but-unused `adjusted_rand_score`, `normalized_mutual_info_score` and `linear_sum_assignment` into actual metrics, and adds a modularity score. `set_graph_for_modularity` — which the training loop still calls — lives there as well. Training does not invoke any of the new metrics.

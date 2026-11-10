# 双曲原型图聚类（拆分版）

将原 `main.py` 按职责拆成若干模块，**训练逻辑与数值行为逐行保持不变**：数据读取与特征平滑、随机游走采样子图、双曲流形与损失、评估各自独立，`main.py` 只保留顶部参数配置与训练流程。

## 1. 目录结构

| 文件 | 职责 |
|---|---|
| `main.py` | 顶部参数配置 + `train_adaptive_pipeline` 训练流程（warmup / 主训练 / 原型初始化 / 保存） |
| `data_preprocess.py` | 数据集读取（OGB / MAG-Scholar-C / Amazon / Planetoid）、标签与边索引规范化、度数统计、特征平滑 |
| `graph_sampling.py` | `GraphStorage`（CSR 全图存储 + 诱导子图抽取）、`weighted_random_walk_partition`（MMRW 采样）、`build_batches` |
| `hyperbolic.py` | `SafePoincareSpace`（庞加莱球 exp/log 映射与距离）、成员度门控、`HyperbolicPrototypeEncoder`、三个损失 + 平衡正则 |
| `evaluation.py` | 模块度全局统计、`modularity_score`、ARI/NMI/ACC、预测硬分配、`evaluate_model`，带命令行入口 |
| `utils.py` | `set_seed`、`save_checkpoint`、`load_checkpoint` |
| `resource_monitor.py` | 训练资源监控（CPU / 内存 / GPU 利用率、显存、功耗、温度），可选 |
| `requirements.txt` | 环境依赖 |

## 2. 环境安装

```bash
pip install -r requirements.txt
```

依赖关键版本：`torch==2.2.2+cu118`、`torch_geometric==2.4.0`、`torch_sparse==0.6.18+pt22cu118`、`ogb==1.3.6`、`scikit-learn==1.7.2`、`numba==0.67.0`。

`torch_sparse` 与 `torch_scatter` 需与 torch / CUDA 版本匹配，若 pip 直接安装失败，按 PyG 官方 wheel 源安装：

```bash
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
```

`numba` 可选：缺失时随机游走的累积分布会退化为纯 Python 循环（结果相同，仅变慢）。

## 3. 数据准备

| 数据集 | 路径 | 说明 |
|---|---|---|
| `ogbn-arxiv` / `ogbn-products` | `{save_path}/data/ogb` | 首次运行自动下载 |
| `cora` / `citeseer` | `{save_path}/data/{name}` | 自动下载 |
| `photo` / `computers` | `{save_path}/data/{name}` | 自动下载 |
| `mag_scholar_c` | `{save_path}/data/mag_scholar_c` | 需手工放置 `features.npz`（稀疏特征）、`edge_index.pt`、`labels.pt` |

`save_path` 在 `main.py` 顶部配置，默认 `/root/autodl-tmp`。MAG-Scholar-C 会先做 128 维 TruncatedSVD 再 L1 归一化。

## 4. 运行

```bash
python main.py
```

改数据集只需修改 `main.py` 顶部两行：

```python
dataset_name = "ogbn-products".lower()
is_large_graph = True          # 大图走 MMRW 采样；cora / citeseer / photo / computers 设为 False
```

训练产物：
- `{SAVE_DIR}/{dataset_name}.pt` —— 用于「已有检查点则跳过训练」的缓存
- `{SAVE_DIR}/{dataset_name}/final.pt` —— 训练结束保存的最终模型

`SAVE_DIR` 默认 `./checkpoints`。

## 5. 配置说明（`main.py` 顶部）

### 5.1 运行级参数

| 变量 | 默认 | 作用 |
|---|---|---|
| `save_path` | `/root/autodl-tmp` | 数据集根目录 |
| `SAVE_DIR` | `./checkpoints` | 检查点输出目录 |
| `seed` | `1` | 全局随机种子（Python / NumPy / Torch），同时用于随机游走 RNG 与 KMeans |
| `dataset_name` | `ogbn-products` | 数据集名，决定下方超参分支 |
| `is_large_graph` | `True` | `True`：MMRW 采样子图训练；`False`：全图训练 |
| `use_ema_prototype` | `= is_large_graph` | `True`：原型做 EMA 更新（缓冲变量）；`False`：原型为可学习参数 |
| `lambda_bal` | `0.05` | 簇平衡 KL 损失权重 |
| `lambda_rec` | `0.05` | 图重构损失权重 |
| `k_steps` | `2` | 特征平滑迭代步数（`0` 表示不平滑） |
| `alpha` | `0.5` | 平滑残差系数：`h ← alpha*h + (1-alpha)*A_norm*h` |
| `ema_momentum` | 分支内赋值 | 原型 EMA 动量（`ogbn-products` 为 0.995） |

### 5.2 数据集分支超参

| 变量 | 作用 |
|---|---|
| `epochs` | 总训练轮数 |
| `warmup_epochs` | 仅用重构损失的预热轮数；结束时用 KMeans 初始化原型 |
| `lr_warmup` / `wd_warmup` | 预热阶段 Adam 学习率 / 权重衰减 |
| `lr_main` / `wd_main` | 主阶段 AdamW 学习率 / 权重衰减 |
| `eta_min_ratio` | 余弦退火下限 = `lr_main * eta_min_ratio` |
| `hidden_dim` | 编码器输出维度（即切空间维度） |
| `middle_dim` | 编码器隐层宽度 |
| `drop_out` | 编码器 Dropout 概率 |
| `WALK_ROOTS_PER_BATCH` | 每个 batch 的随机游走根节点数 |
| `WALK_NUM_BATCHES` | batch 数量 |
| `WALK_T_PER_ROOT` | 每个根节点出发的游走次数 |
| `WALK_DEPTH` | 单次游走长度 |
| `WALK_TAU_FEAT` | 特征相似度权重温度：`exp(cos/tau)` |
| `WALK_MAX_BATCH` / `WALK_MIN_BATCH` | 单 batch 节点数上限 / 下限（低于下限的 batch 丢弃） |

现有分支取值：

| 数据集 | epochs | warmup | lr_warmup | lr_main | hidden | drop_out |
|---|---|---|---|---|---|---|
| `ogbn-arxiv` | 180 | 30 | 1e-3 | 5e-4 | 128 | 0.1 |
| `ogbn-products` | 180 | 80 | 1e-2 | 5e-4 | 128 | 0.1 |
| `citeseer` | 400 | 150 | 1e-4 | 5e-5 | 256 | 0.5 |
| `cora` | 180 | 80 | 5e-4 | 5e-5 | 64 | 0.1 |
| `photo` | 350 | 100 | 3e-3 | 4e-5 | 256 | 0.0 |
| `computers` | 500 | 150 | 5e-3 | 4e-5 | 256 | 0.1 |
| `mag_scholar_c` | 200 | 50 | 1e-2 | 5e-4 | 128 | 0.1 |

新增数据集：在分支链里补一段赋值即可，或在 `data_preprocess.load_data` 中补对应的读取分支。

> 注意：`WALK_*` 只在 `ogbn-arxiv` / `ogbn-products` / `mag_scholar_c` 三个分支里定义。若在小图数据集上把 `is_large_graph` 设为 `True`，这些变量不存在会报 `NameError`；反之在大图数据集上设为 `False` 不受影响（这是原代码的既有行为，未改动）。

### 5.3 检查点行为

```python
model, data, y_true = train_adaptive_pipeline(
    dataset_name=dataset_name, hidden_dim=hidden_dim,
    epochs=epochs, warmup_epochs=warmup_epochs,
    is_large_graph=is_large_graph,
    load_pretrained=False,   # True 且缓存存在时，直接加载并 return，不训练
    force_retrain=True,      # True 强制重新训练
)
```

即 `load_pretrained=True and force_retrain=False` 且 `{SAVE_DIR}/{dataset_name}.pt` 存在时，跳过训练直接返回模型。

## 6. 评估

`evaluation.py` 可单独作为脚本运行（从检查点自动推断 `in_features / middle_dim / hidden_dim / 簇数 / 是否 EMA 原型`，无需手工填超参）：

```bash
# 全图数据集
python evaluation.py --dataset-name cora \
    --checkpoint ./checkpoints/cora/final.pt \
    --data-root /root/autodl-tmp

# 大图（按采样方式推理，再用全图边计算模块度）
python evaluation.py --dataset-name ogbn-products --is-large-graph \
    --checkpoint ./checkpoints/ogbn-products/final.pt
```

输出示例：

```json
{
  "n_evaluated": 232965,
  "ARI": 0.4123,
  "NMI": 0.5012,
  "ACC": 0.6431,
  "Modularity": 0.5871
}
```

- `ARI` / `NMI`：与真实标签的聚类一致性；`ACC`：匈牙利匹配后的最优映射准确率。
- `Modularity`：以全图缓存的度数统计与 `2m` 计算的 Newman 模块度，与训练时损失中的 `global_deg / global_m2` 定义一致。
- 大图模式下节点按 batch 推理，未落入任何 batch 的节点记为 `-1`，不计入指标。
- 大图的采样参数默认对齐 `ogbn-products`，可用 `--roots-per-batch / --num-batches / --t-per-root / --walk-depth / --tau-feat / --max-batch / --min-batch` 覆盖。
- 其他参数：`--k-steps`、`--alpha`（必须与训练时一致）、`--drop-out`、`--ema-momentum`、`--device`、`--data-root`、`--save-dir`。

评估为独立工具，**训练流程本身不调用它**，因此不影响原有训练逻辑。

## 7. 训练资源监控（可选）

```python
from resource_monitor import ResourceMonitor

monitor = ResourceMonitor(interval=1.0)
monitor.start()

# 训练 ...

monitor.stop()
monitor.report()          # 打印汇总
summary = monitor.summary()   # 结构化 dict，便于落盘
```

监控 CPU 利用率、系统内存、每张 GPU 的利用率 / 显存 / 功耗 / 温度，以及 PyTorch 的峰值显存。

## 8. 模块接口速查

```python
# 数据
from data_preprocess import prepare_data
data, num_classes, y_true, num_nodes, global_deg, global_m2 = prepare_data(
    dataset_name, save_path, k_steps, alpha)
# 平滑特征在 data.x_input

# 采样
from graph_sampling import GraphStorage, weighted_random_walk_partition, build_batches
graph_storage = GraphStorage(data.edge_index, num_nodes, data.x_input, device)
batches = weighted_random_walk_partition(graph_storage, roots_per_batch=..., seed=seed)
batch_data = build_batches(graph_storage, batches, data.x_input)   # [{node_ids, x, edge_index, num_nodes}]

# 模型与损失
from hyperbolic import HyperbolicPrototypeEncoder, compute_graph_rec_loss, \
     hyperbolic_modularity_loss, scalable_hsc_loss, balance_loss
model = HyperbolicPrototypeEncoder(in_features, hidden_dim, num_clusters,
        middle_dim=middle_dim, drop_out=drop_out,
        use_ema_prototype=..., ema_momentum=..., is_large_graph=..., seed=seed)

# 评估
from evaluation import evaluate_model
report = evaluate_model(model, y_true, num_classes=num_classes,
        x_input=data.x_input, edge_index=data.edge_index,
        num_nodes=num_nodes, device=device)
```

## 9. 关于本次拆分

- 所有被搬移的函数 / 类，函数体与原文逐行一致（仅新增英文 docstring）。
- 原代码依赖 `main.py` 模块级全局变量 `middle_dim`、`drop_out`、`is_large_graph`、`seed` 的地方，改为显式传参（`HyperbolicPrototypeEncoder` 构造参数），数值来源仍是顶部配置，计算结果不变。
- 平滑特征后删除 `data.x`、`prepare_data` 中被丢弃的中间张量、`gc.collect()` 调用位置，均与原文一致，峰值内存行为不变。
- 全部中文注释已移除并改写为英文注释 / docstring；运行时打印的字符串保持原样（含中文），以保证输出与原来一致。若需要一并英文化，直接改 `print` 文案即可，不影响逻辑。

# Qwen3-0.6B 从零训练指南

本项目旨在从零开始训练一个 Qwen3-0.6B 模型，并进行 MLP 中间层大小的消融实验。

## 已完成步骤

1.  **环境配置**:
    *   激活了 `oracle-moe` Conda 环境。
    *   设置了环境变量 `HF_ENDPOINT=https://hf-mirror.com` 以加速下载。
    *   配置了分布式训练相关的环境变量 (`NCCL_DEBUG`, `NCCL_IB_DISABLE`)。

2.  **配置下载**:
    *   从 `Qwen/Qwen3-0.6B` 下载了配置文件到 `/data/250010109/qwen_exp/configs`。

3.  **模型代码准备**:
    *   从 `transformers` 库中提取了 Qwen3 的建模代码，并修复了相对导入路径以支持本地调用。
    *   存放在 `/data/250010109/qwen_exp/src/model/qwen3`。

4.  **训练脚本优化与实验支持**:
    *   **修复错误**: 解决了 DeepSpeed ZeRO-1 模式下未显式提供优化器的 `AssertionError`。
    *   **参数量统计**: 脚本现在在训练开始前打印模型总参数量和可训练参数量。
    *   **动态 MLP 比例**: 添加了 `--mlp_ratio` 参数，允许动态调整 `intermediate_size`。
    *   **自动化实验**: 更新了 `run_train.sh`，支持对 `mlp_ratio` 在 `[0.5, 1.0, 2.0, 4.0]` 范围内进行自动化实验。

## 实验设置 (MLP 比例实验)

对于 `hidden_size=1024` 的 Qwen3-0.6B 模型，实验设置如下：
- **Ratio 0.5**: `intermediate_size=512`, 输出目录: `output/qwen3-0.6b-mlp-512`
- **Ratio 1.0**: `intermediate_size=1024`, 输出目录: `output/qwen3-0.6b-mlp-1024`
- **Ratio 2.0**: `intermediate_size=2048`, 输出目录: `output/qwen3-0.6b-mlp-2048`
- **Ratio 4.0**: `intermediate_size=4096`, 输出目录: `output/qwen3-0.6b-mlp-4096`

## 训练策略与日志风格

本项目已参考 `externel_resources/pretrain_bin.py` 优化了训练策略和打印风格：

1.  **学习率调度器**: 
    - 使用带 Warmup 的余弦退火学习率调度器 (`CosineAnnealingLR with Warmup`)。
    - 调度参数基于总训练步数动态计算。
2.  **训练数据量限制**: 
    - 设定总训练数据量为 **10B tokens**。
    - 脚本会实时统计已处理的 tokens 数，达到目标后自动停止。
3.  **日志打印风格**:
    - **启动信息**: 打印详细的 GPU、批次大小、Token 统计等配置。
    - **实时监控**: 
        - 使用 `tqdm` 进度条显示 `loss`、`lr` 和 `opt_step`。
        - 每 10 步打印详细日志，包含 `batch (opt_step-local_step)`、`loss`、`lr`、`batch_time` 和 `total_tokens`。
    - **准确统计**: 跨卡同步统计总 tokens 数，确保数据量计算精确。
4.  **断点续训优化**: 
    - 改进了 `resume` 逻辑，基于 `samples_seen` 准确跳过已训练样本，支持不同卡数下的稳定恢复。

### 训练超参数建议

- **Batch Size**: 
    - 当前设置为 `384` (约 400K tokens/step)。
    - **风险提示**: 如果发现 Loss 不收敛或泛化差，建议降至 `128` 或 `256`。大 Batch 虽然跑得快，但对小模型可能存在“学习不透”的问题。
- **Learning Rate**: 
    - 配合 400K Batch Size，`1e-4` 是一个保守且安全的起点。
    - 如果 Batch Size 缩小，可以适当调低 LR。

## 严谨消融实验：MLP 宽度 Scaling 分析 (Fixed Depth)

为了深入探究 MoE 专家的参数效率，我们设计了**固定深度、变量宽度**的消融实验。

- **核心假设**: 如果我们能证明较小的 MLP 宽度（如 $k=1.0$）在固定深度下会显著降低模型性能，那么后续引入 MoE（通过多专家恢复总参数量，同时保持计算量）的价值就得到了证明。
- **实验设计**:
    - **固定层数 (L)**: 28 层（与 Qwen3-0.6B 原厂一致）。
    - **变量系数 (k)**: $[0.5, 1.0, 2.0, 3.0(Base), 4.0]$。
    - **观测指标**: 10B Tokens 训练后的验证集 Loss、收敛速度、以及参数 ROI（Return on Investment）。

### 如何启动训练

直接指定 MLP 比例即可，层数将固定为 28：

```bash
# 运行 0.5 比例实验 (窄 MLP)
bash run_train.sh 0.5

# 运行 3.0 比例实验 (原厂配置)
bash run_train.sh 3.0

# 运行 4.0 比例实验 (宽 MLP)
bash run_train.sh 4.0
```

## 监控

- **日志**: 每个实验的日志位于其对应的 `logs` 目录下。
- **TensorBoard**: 可视化数据同样位于 `logs` 目录，可通过 TensorBoard 启动查看。

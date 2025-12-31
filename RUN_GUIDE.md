# Qwen3-0.6B 训练运行指南

## 实验说明

本项目包含三个消融实验：

### 实验 1: MLP 宽度 Scaling（固定深度）
- **变量**: MLP ratio (intermediate_size / hidden_size)
- **固定**: 层数 L=28
- **目的**: 观察 MLP 宽度对模型性能的影响
- **参数量**: 随 ratio 变化

### 实验 2: MLP 宽度 + 深度 Scaling（参数量恒定）
- **变量**: MLP ratio + 层数 L
- **固定**: 总参数量
- **目的**: 通过调节层数保持参数量恒定
- **参数量**: 恒定

### 实验 3: MLP 宽度 + LoRA Scaling（固定深度）
- **变量**: MLP ratio + LoRA rank
- **固定**: 层数 L=28，总参数量恒定
- **目的**: 通过 LoRA 替换部分参数保持参数量恒定
- **参数量**: 恒定

---

## LoRA 模式说明

| 模式 | Base Linear | LoRA Linear | 训练方式 | 参数量计算 |
|------|-------------|--------------|----------|-------------|
| `lora` | ❌ 不存在 | ✅ 存在 | 只有 LoRA 可训练 | 仅 LoRA 参数 |
| `lora_bias` | ✅ 存在（冻结） | ✅ 存在 | 只有 LoRA 可训练 | Base + LoRA 参数 |

### 初始化策略

**LoRA A 矩阵**: Normal(mean=0, std=1/rank)
- 工业界常用的 LoRA 初始化方式
- std 与 rank 成反比，保证初始值较小

**LoRA B 矩阵**: Zeros
- 保证训练开始时 LoRA 输出接近零
- 模型初始行为接近 base 模型

**Base 权重** (lora_bias 模式): Kaiming Uniform
- PyTorch 默认的线性层初始化
- a=5^0.5，适合 ReLU/SiLU 激活函数

---

## 目录结构

```
/data/250010109/qwen_exp/
├── configs/                          # 模型配置文件
│   ├── config.json                 # Qwen3-0.6B 基础配置
│   ├── tokenizer_config.json
│   └── generation_config.json
├── src/
│   ├── model/qwen3/
│   │   ├── modeling_qwen3.py      # 模型实现
│   │   └── configuration_qwen3.py # 配置类
│   └── data_utils.py               # 数据加载
├── train_qwen3.py                   # 训练脚本
├── run_train.sh                     # 实验 1 & 2 训练脚本
├── run_train_lora.sh                # 实验 3 训练脚本
├── verify_lora_params.py             # LoRA 参数量验证
└── verify_iso_params.py              # ISO 参数量验证
```

---

## 输出目录结构

所有训练输出保存在 `/data/share/109_cache_dir/qwen_exp/output/`

```
/data/share/109_cache_dir/qwen_exp/output/
├── qwen3-0.6b-ratio-3.0-nolora/              # Baseline (无 LoRA)
│   ├── checkpoints/
│   │   └── checkpoint-{step}/
│   │       ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │       ├── ...
│   │       └── latest
│   └── logs/
│       └── train_YYYYMMDD_HHMMSS.log
├── qwen3-0.6b-ratio-0.5-nolora/           # 实验 1: ratio=0.5
├── qwen3-0.6b-ratio-1.0-nolora/           # 实验 1: ratio=1.0
├── qwen3-0.6b-ratio-2.0-nolora/           # 实验 1: ratio=2.0
├── qwen3-0.6b-ratio-4.0-nolora/           # 实验 1: ratio=4.0
├── qwen3-0.6b-ratio-0.25-lora-rank2458-lora/      # 实验 3: ratio=0.25 + LoRA
│   ├── checkpoints/
│   │   └── checkpoint-{step}/
│   └── logs/
│       └── train_YYYYMMDD_HHMMSS.log
├── qwen3-0.6b-ratio-0.5-lora-rank2048-lora/       # 实验 3: ratio=0.5 + LoRA
├── qwen3-0.6b-ratio-1.0-lora-rank1536-lora/       # 实验 3: ratio=1.0 + LoRA
├── qwen3-0.6b-ratio-0.25-lora-rank2458-lora_bias/ # 实验 3: ratio=0.25 + LoRA_bias
├── qwen3-0.6b-ratio-0.5-lora-rank2048-lora_bias/ # 实验 3: ratio=0.5 + LoRA_bias
└── qwen3-0.6b-ratio-1.0-lora-rank1536-lora_bias/ # 实验 3: ratio=1.0 + LoRA_bias
```

### 日志文件位置

日志文件位于各实验目录的 `logs/` 子目录中：
- 文件名格式: `train_YYYYMMDD_HHMMSS.log`
- 包含完整的训练输出
- 可通过 `tail -f` 实时查看

### Checkpoint 位置

Checkpoints 位于各实验目录的 `checkpoints/` 子目录中：
- 目录名格式: `checkpoint-{step}`
- 包含模型权重、优化器状态等
- 可用于断点续训

---

## 运行方式

### 实验 1: MLP 宽度 Scaling（无 LoRA）

```bash
# Baseline (ratio=3.0, 无 LoRA)
bash run_train.sh 3.0

# 实验 1 配置
bash run_train.sh 0.5
bash run_train.sh 1.0
bash run_train.sh 2.0
bash run_train.sh 4.0
```

### 实验 2: MLP 宽度 + 深度 Scaling（ISO 参数量）

```bash
# 使用 --iso_parameter 参数自动调整层数
bash run_train.sh 0.5 --iso_parameter
bash run_train.sh 1.0 --iso_parameter
```

### 实验 3: MLP 宽度 + LoRA Scaling

```bash
# 用法: bash run_train_lora.sh <mlp_ratio> <lora_rank> <lora_mode>

# 3.1 LoRA 模式（只有 LoRA，无 base）
bash run_train_lora.sh 0.25 2458 lora
bash run_train_lora.sh 0.5 2048 lora
bash run_train_lora.sh 1.0 1536 lora

# 3.2 LoRA_bias 模式（base 冻结 + LoRA）
bash run_train_lora.sh 0.25 2458 lora_bias
bash run_train_lora.sh 0.5 2048 lora_bias
bash run_train_lora.sh 1.0 1536 lora_bias
```

---

## 参数量验证

在运行实验前，可以使用验证脚本确认参数量：

```bash
# 验证 LoRA 参数量
python verify_lora_params.py

# 验证 ISO 参数量
python verify_iso_params.py
```

输出示例：
```
Baseline (无 LoRA): ratio=3.0, Total Params = 596.05M

Ratio    Intermediate Mode         Rank     Actual Total    Trainable    Diff %
0.25     256          lora         2458     596.09M         596.09M      +0.01%
0.5      512          lora         2048     596.05M         596.05M      +0.00%
1.0      1024         lora         1536     596.05M         596.05M      +0.00%
```

---

## 监控训练

### 查看实时日志

```bash
# 替换为实际的实验目录
tail -f /data/share/109_cache_dir/qwen_exp/output/qwen3-0.6b-ratio-0.25-lora-rank2458-lora/logs/train_*.log
```

### 查看 TensorBoard

```bash
# 替换为实际的实验目录
tensorboard --logdir=/data/share/109_cache_dir/qwen_exp/output/qwen3-0.6b-ratio-0.25-lora-rank2458-lora/logs
```

---

## 断点续训

如果训练中断，可以使用 `--resume` 参数继续训练：

```bash
# 修改 run_train_lora.sh，在 deepspeed 命令中添加:
# --resume

deepspeed --num_gpus=${NUM_GPUS} --master_port=${MASTER_PORT} train_qwen3.py \
    ... \
    --resume \
    ...
```

---

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `SEQ_LEN` | 1024 | 序列长度 |
| `LR` | 1e-4 | 学习率 |
| `BATCH_SIZE` | 8 | 单卡 batch size |
| `GLOBAL_BATCH_SIZE` | 512 | 全局 batch size |
| `SAVE_INTERVAL` | 1000 | 保存间隔 |
| `TOTAL_TOKENS` | 10000000000 | 总训练 tokens (10B) |
| `WARMUP_STEPS` | 2000 | Warmup 步数 |
| `ZERO_STAGE` | 1 | DeepSpeed Zero 阶段 |

---

## 实验配置汇总

| 实验组 | mlp_ratio | LoRA Rank | LoRA Mode | 参数量 | Model Name |
|--------|-----------|-------------|-------------|--------|------------|
| Baseline | 3.0 | 0 | none | 596.05M | qwen3-0.6b-ratio-3.0-nolora |
| Exp1-0.5 | 0.5 | 0 | none | ~669M | qwen3-0.6b-ratio-0.5-nolora |
| Exp1-1.0 | 1.0 | 0 | none | ~713M | qwen3-0.6b-ratio-1.0-nolora |
| Exp1-2.0 | 2.0 | 0 | none | ~801M | qwen3-0.6b-ratio-2.0-nolora |
| Exp3-0.25 | 0.25 | 2458 | lora | 596.09M | qwen3-0.6b-ratio-0.25-lora-rank2458-lora |
| Exp3-0.5 | 0.5 | 2048 | lora | 596.05M | qwen3-0.6b-ratio-0.5-lora-rank2048-lora |
| Exp3-1.0 | 1.0 | 1536 | lora | 596.05M | qwen3-0.6b-ratio-1.0-lora-rank1536-lora |
| Exp3-0.25 | 0.25 | 2458 | lora_bias | 618.11M | qwen3-0.6b-ratio-0.25-lora-rank2458-lora_bias |
| Exp3-0.5 | 0.5 | 2048 | lora_bias | 640.09M | qwen3-0.6b-ratio-0.5-lora-rank2048-lora_bias |
| Exp3-1.0 | 1.0 | 1536 | lora_bias | 684.13M | qwen3-0.6b-ratio-1.0-lora-rank1536-lora_bias |

---

## 常见问题

### Q1: 如何确定 LoRA rank？
A1: 运行 `python verify_lora_params.py` 查看 "Rank Needed" 列。

### Q2: lora 和 lora_bias 模式如何选择？
A2:
- `lora`: 参数量最少，只有 LoRA 部分计入总参数量
- `lora_bias`: 参数量更多（包含冻结的 base），但可能收敛更好

### Q3: 如何检查训练是否正常运行？
A3: 查看日志文件中的 loss 值是否正常下降，lr 是否按 schedule 变化。

### Q4: 磁盘空间不足怎么办？
A4: 减小 `SAVE_INTERVAL` 或删除旧实验的 checkpoint 目录。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `train_qwen3.py` | 主训练脚本 |
| `run_train.sh` | 实验 1 & 2 运行脚本 |
| `run_train_lora.sh` | 实验 3 运行脚本 |
| `verify_lora_params.py` | LoRA 参数量验证 |
| `verify_iso_params.py` | ISO 参数量验证 |
| `RUN_GUIDE.md` | 本文档 |

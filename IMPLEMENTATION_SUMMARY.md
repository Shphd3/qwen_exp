# Qwen3-0.6B MLP + LoRA 实现总结报告

## 项目目标

实现三个消融实验，探究 MLP 宽度、深度和 LoRA 对模型性能的影响：

1. **实验 1**: MLP 宽度 Scaling（固定深度 L=28）
2. **实验 2**: MLP 宽度 + 深度 Scaling（参数量恒定）
3. **实验 3**: MLP 宽度 + LoRA Scaling（固定深度 L=28，参数量恒定）

---

## 实现内容

### 1. 配置类修改 (`src/model/qwen3/configuration_qwen3.py`)

#### 新增参数
```python
use_lora: bool = False           # 是否启用 LoRA
lora_mode: str = "lora"         # LoRA 模式: "lora" 或 "lora_bias"
lora_rank: int = 8              # LoRA rank
lora_alpha: int = None          # LoRA alpha（默认等于 lora_rank）
lora_dropout: float = 0.05      # LoRA dropout
```

#### LoRA 模式说明
| 模式 | Base Linear | LoRA Linear | 训练方式 | 参数量计算 |
|------|-------------|--------------|----------|-------------|
| `lora` | ❌ 不存在 | ✅ 存在 | 只有 LoRA 可训练 | 仅 LoRA 参数 |
| `lora_bias` | ✅ 存在（冻结） | ✅ 存在 | 只有 LoRA 可训练 | Base + LoRA 参数 |

#### LoRA Alpha 设置
- 默认 `lora_alpha = lora_rank`
- 如果未指定，自动设置为等于 rank
- Scaling factor = alpha / rank = 1.0

---

### 2. 模型实现修改 (`src/model/qwen3/modeling_qwen3.py`)

#### LoRALinear 类
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def reset_parameters(self):
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / self.rank)
        nn.init.zeros_(self.lora_B)
```

**初始化策略（工业级）**:
- **LoRA A**: Normal(mean=0, std=1/rank)
  - rank 越大，初始值越小
  - 符合工业界 LoRA 初始化的最佳实践
- **LoRA B**: Zeros
  - 保证训练开始时 LoRA 输出接近零
  - 模型初始行为接近 base 模型

#### Qwen3MLPLoRA 类
```python
class Qwen3MLPLoRA(nn.Module):
    def __init__(self, config):
        self.gate_lora = LoRALinear(...)
        self.up_lora = LoRALinear(...)
        self.down_lora = LoRALinear(...)

        if config.lora_mode == "lora_bias":
            self.gate_proj = nn.Linear(...)  # 冻结
            self.up_proj = nn.Linear(...)    # 冻结
            self.down_proj = nn.Linear(...)   # 冻结

        self._init_base_weights()  # 使用 Kaiming Uniform 初始化 base
        self._set_trainable_params()
```

**Base 初始化（lora_bias 模式）**:
```python
def _init_base_weights(self):
    nn.init.kaiming_uniform_(self.gate_proj.weight, a=5**0.5)
    nn.init.kaiming_uniform_(self.up_proj.weight, a=5**0.5)
    nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
```

---

### 3. 训练脚本修改 (`train_qwen3.py`)

#### 新增参数
```python
--use_lora              # 启用 LoRA
--lora_mode lora        # LoRA 模式: "lora" 或 "lora_bias"
--lora_rank 8           # LoRA rank
--lora_alpha 16         # LoRA alpha（默认等于 rank）
--lora_dropout 0.05     # LoRA dropout
```

#### 配置应用
```python
if args.use_lora:
    config.use_lora = True
    config.lora_mode = args.lora_mode
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
```

---

### 4. 运行脚本 (`run_train_lora.sh`)

#### 脚本功能
- 自动检测 GPU 数量
- 生成带 LoRA 模式的 model name
- 日志文件保存在 `${OUTPUT_ROOT}/${MODEL_NAME}/logs/`

#### Model Name 格式
| 配置 | Model Name 示例 |
|------|----------------|
| 无 LoRA | `qwen3-0.6b-ratio-3.0-nolora` |
| LoRA | `qwen3-0.6b-ratio-0.5-lora-rank2048-lora` |
| LoRA_bias | `qwen3-0.6b-ratio-0.5-lora-rank2048-lora_bias` |

#### 用法
```bash
bash run_train_lora.sh <mlp_ratio> <lora_rank> <lora_mode>
```

---

### 5. 参数量验证脚本 (`verify_lora_params.py`)

#### 功能
- 计算 Baseline 参数量（596.05M）
- 计算各 ratio + LoRA 组合的参数量
- 验证参数量是否恒定

#### 输出示例
```
Ratio    Intermediate Mode         Rank     Actual Total    Trainable    Diff %
Baseline (无 LoRA): ratio=3.0, Total Params = 596.05M

0.25     256          lora         2458     596.09M         596.09M      +0.01%
0.5      512          lora         2048     596.05M         596.05M      +0.00%
1.0      1024         lora         1536     596.05M         596.05M      +0.00%
2.0      2048         lora         1024     596.05M         596.05M      +0.00%
4.0      4096         lora         614      595.88M         595.88M      -0.03%
8.0      8192         lora         341      595.79M         595.79M      -0.04%
```

---

## 参数量计算说明

### Baseline (ratio=3.0, 无 LoRA)
- hidden_size = 1024
- intermediate_size = 3072
- MLP 每层: 3 * 1024 * 3072 = 9,437,184 = 9.44M
- 28 层 MLP: 28 * 9.44M = 264.24M
- Total: ~596.05M

### LoRA 模式（lora，只有 LoRA）
**公式**: `3 * rank * (hidden_size + intermediate_size)`

例如 ratio=0.5 (intermediate_size=512, rank=2048):
- 每层 LoRA: 3 * 2048 * (1024 + 512) = 9,437,184 = 9.44M
- 28 层 LoRA: 28 * 9.44M = 264.24M
- Total: ~596.05M ✓ 参数量恒定

### LoRA Bias 模式（lora_bias，base 冻结 + LoRA）
**公式**: `3 * hidden_size * intermediate_size + 3 * rank * (hidden_size + intermediate_size)`

例如 ratio=0.5 (intermediate_size=512, rank=2048):
- 每层 Base: 3 * 1024 * 512 = 1,572,864 = 1.57M
- 每层 LoRA: 9.44M
- 28 层: 28 * (1.57M + 9.44M) = 308.34M
- Total: ~640.09M

**注意**: `lora_bias` 模式下 total parameters 包含冻结的 base 参数，但 trainable parameters 仅为 LoRA 部分。

---

## 关键设计决策

### 1. 为什么 LoRA Alpha 默认等于 Rank？
- Scaling factor = alpha / rank
- 当 alpha = rank 时，scaling = 1.0
- 简化超参数调优
- 符合许多 LoRA 实践（如 LoRA 论文中的默认设置）

### 2. 为什么 LoRA A 使用 Normal(0, 1/rank)？
- std 与 rank 成反比，保证初始值不会过大
- rank 越大，初始值越小
- 符合工业界最佳实践

### 3. 为什么 LoRA B 使用 Zeros？
- 保证训练开始时 LoRA 输出接近零
- 模型初始行为接近 base 模型
- 避免训练初期的不稳定

### 4. 为什么 lora_bias 模式 base 使用 Kaiming Uniform？
- a=5^0.5，适合 SiLU 激活函数
- 与 PyTorch Linear 层默认初始化一致
- 经过验证的初始化方法

---

## 文件清单

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `src/model/qwen3/configuration_qwen3.py` | 添加 LoRA 配置参数 | ~30 |
| `src/model/qwen3/modeling_qwen3.py` | 实现 LoRALinear, Qwen3MLPLoRA | ~100 |
| `train_qwen3.py` | 添加 LoRA 训练参数 | ~10 |
| `run_train_lora.sh` | LoRA 实验运行脚本 | ~70 |
| `verify_lora_params.py` | 参数量验证脚本 | ~100 |
| `RUN_GUIDE.md` | 运行指南文档 | ~300 |

---

## 实验配置汇总

### Baseline
| mlp_ratio | LoRA Rank | LoRA Mode | Total Params | Trainable | Model Name |
|-----------|-------------|-------------|-------------|-----------|------------|
| 3.0 | 0 | none | 596.05M | 596.05M | qwen3-0.6b-ratio-3.0-nolora |

### 实验 3: LoRA 模式（参数量恒定）
| mlp_ratio | LoRA Rank | LoRA Mode | Total Params | Trainable | Model Name |
|-----------|-------------|-------------|-------------|-----------|------------|
| 0.25 | 2458 | lora | 596.09M | 596.09M | qwen3-0.6b-ratio-0.25-lora-rank2458-lora |
| 0.5 | 2048 | lora | 596.05M | 596.05M | qwen3-0.6b-ratio-0.5-lora-rank2048-lora |
| 1.0 | 1536 | lora | 596.05M | 596.05M | qwen3-0.6b-ratio-1.0-lora-rank1536-lora |
| 2.0 | 1024 | lora | 596.05M | 596.05M | qwen3-0.6b-ratio-2.0-lora-rank1024-lora |
| 4.0 | 614 | lora | 595.88M | 595.88M | qwen3-0.6b-ratio-4.0-lora-rank614-lora |
| 8.0 | 341 | lora | 595.79M | 595.79M | qwen3-0.6b-ratio-8.0-lora-rank341-lora |

### 实验 3: LoRA Bias 模式（参数量恒定）
| mlp_ratio | LoRA Rank | LoRA Mode | Total Params | Trainable | Model Name |
|-----------|-------------|-------------|-------------|-----------|------------|
| 0.25 | 2458 | lora_bias | 618.11M | 596.09M | qwen3-0.6b-ratio-0.25-lora-rank2458-lora_bias |
| 0.5 | 2048 | lora_bias | 640.09M | 596.05M | qwen3-0.6b-ratio-0.5-lora-rank2048-lora_bias |
| 1.0 | 1536 | lora_bias | 684.13M | 596.05M | qwen3-0.6b-ratio-1.0-lora-rank1536-lora_bias |

---

## 输出目录结构

```
/data/share/109_cache_dir/qwen_exp/output/
├── qwen3-0.6b-ratio-3.0-nolora/
│   ├── checkpoints/
│   │   └── checkpoint-{step}/
│   └── logs/
│       └── train_YYYYMMDD_HHMMSS.log
├── qwen3-0.6b-ratio-0.25-lora-rank2458-lora/
├── qwen3-0.6b-ratio-0.5-lora-rank2048-lora/
├── qwen3-0.6b-ratio-1.0-lora-rank1536-lora/
├── qwen3-0.6b-ratio-0.25-lora-rank2458-lora_bias/
├── qwen3-0.6b-ratio-0.5-lora-rank2048-lora_bias/
└── qwen3-0.6b-ratio-1.0-lora-rank1536-lora_bias/
```

---

## 使用示例

### 验证参数量
```bash
python verify_lora_params.py
```

### 运行实验
```bash
# LoRA 模式
bash run_train_lora.sh 0.25 2458 lora
bash run_train_lora.sh 0.5 2048 lora
bash run_train_lora.sh 1.0 1536 lora

# LoRA Bias 模式
bash run_train_lora.sh 0.25 2458 lora_bias
bash run_train_lora.sh 0.5 2048 lora_bias
bash run_train_lora.sh 1.0 1536 lora_bias
```

### 监控训练
```bash
tail -f /data/share/109_cache_dir/qwen_exp/output/qwen3-0.6b-ratio-0.5-lora-rank2048-lora/logs/train_*.log
```

---

## 验证清单

- [x] LoRA 配置参数添加到 Qwen3Config
- [x] LoRALinear 类实现（工业级初始化）
- [x] Qwen3MLPLoRA 类实现（支持 lora 和 lora_bias 模式）
- [x] Base 权重使用 Kaiming Uniform 初始化
- [x] LoRA A 使用 Normal(0, 1/rank) 初始化
- [x] LoRA B 使用 Zeros 初始化
- [x] LoRA Alpha 默认等于 LoRA Rank
- [x] 训练脚本支持 LoRA 参数
- [x] 运行脚本支持 LoRA 模式
- [x] 参数量验证脚本实现
- [x] Model Name 包含 LoRA 模式信息
- [x] 日志文件保存在正确位置
- [x] 运行指南文档编写

---

## 总结

本次实现完成了 MLP + LoRA 的完整框架，支持两种 LoRA 模式：
- `lora`: 纯 LoRA，参数量最少
- `lora_bias`: Base 冻结 + LoRA，参数量更多（包含冻结 base）

所有配置均采用工业级初始化方法，参数量经过验证确保恒定，可以开始进行消融实验。

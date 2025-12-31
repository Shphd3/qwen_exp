import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from src.model.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from src.model.qwen3.configuration_qwen3 import Qwen3Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def find_rank_for_params(target_ratio, base_ratio, base_mlp_params_per_layer, hidden_size=1024, num_layers=28):
    intermediate_size = int(hidden_size * target_ratio)
    
    lora_params_needed_per_layer = base_mlp_params_per_layer
    rank_per_layer = lora_params_needed_per_layer / (3 * (hidden_size + intermediate_size))
    
    return int(round(rank_per_layer))


def verify_config_ratio_lora(ratio, rank, lora_mode="lora"):
    config = Qwen3Config.from_pretrained("/data/250010109/qwen_exp/configs")
    config.intermediate_size = int(config.hidden_size * ratio)
    config.use_lora = True
    config.lora_mode = lora_mode
    config.lora_rank = rank
    
    model = Qwen3ForCausalLM(config)
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    return total_params, trainable_params


def main():
    print("=" * 120)
    print("实验3: LoRA 参数量恒定实验 (L=28, d=[0.25d, 0.5d, 1d, 2.0d, 4.0d, 8.0d])")
    print("=" * 120)

    hidden_size = 1024
    num_layers = 28
    base_ratio = 3.0
    base_intermediate_size = int(hidden_size * base_ratio)

    config_base = Qwen3Config.from_pretrained("/data/250010109/qwen_exp/configs")
    model_base = Qwen3ForCausalLM(config_base)
    base_params = count_parameters(model_base)

    base_mlp_params_per_layer = 3 * hidden_size * base_intermediate_size

    print(f"\nBaseline (无 LoRA): ratio={base_ratio}, Total Params = {format_number(base_params)}")
    print(f"hidden_size={hidden_size}, num_layers={num_layers}, intermediate_size={base_intermediate_size}")
    print(f"Base MLP params per layer: {format_number(base_mlp_params_per_layer)}")
    print()

    ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"{'Ratio':<8} {'Intermediate':<12} {'Mode':<12} {'Rank':<8} {'Actual Total':<15} {'Trainable':<12} {'Diff %':<10}")
    print("-" * 95)

    for ratio in ratios:
        intermediate_size = int(hidden_size * ratio)
        rank = find_rank_for_params(ratio, base_ratio, base_mlp_params_per_layer, hidden_size, num_layers)
        
        print(f"{ratio:<8} {intermediate_size:<12} {'lora':<12} {rank:<8} ", end="")
        
        total_params, trainable_params = verify_config_ratio_lora(ratio, rank, "lora")
        diff = (total_params - base_params) / base_params * 100
        
        print(f"{format_number(total_params):<15} {format_number(trainable_params):<12} {diff:+.2f}%")
        
        if ratio in [0.25, 0.5, 1.0]:
            print(f"{'':<8} {'':<12} {'lora_bias':<12} {rank:<8} ", end="")
            total_params_bias, trainable_params_bias = verify_config_ratio_lora(ratio, rank, "lora_bias")
            diff_bias = (total_params_bias - base_params) / base_params * 100
            print(f"{format_number(total_params_bias):<15} {format_number(trainable_params_bias):<12} {diff_bias:+.2f}%")

    print("\n" + "=" * 120)
    print("参数量计算公式:")
    print("=" * 120)
    print("  Baseline MLP (ratio=3.0): 3 * hidden_size * intermediate_size = 3 * 1024 * 3072 = 9.44M per layer")
    print("  LoRA MLP (ratio=k):      3 * rank * (hidden_size + intermediate_size)")
    print()
    print("  使参数量恒定: rank = base_mlp_params / [3 * (hidden_size + intermediate_size)]")
    print()
    print("  模式说明:")
    print("    - lora:       只有 LoRA 矩阵，无 base 矩阵（base 不计入总参数量）")
    print("    - lora_bias:  使用 base 矩阵（冻结）+ LoRA 矩阵（可训练）")
    print()
    print("=" * 120)


if __name__ == "__main__":
    main()

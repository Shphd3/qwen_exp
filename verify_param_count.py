import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from src.model.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from src.model.qwen3.configuration_qwen3 import Qwen3Config


def calculate_params(model, mode="total"):
    if mode == "total":
        return sum(p.numel() for p in model.parameters())
    elif mode == "trainable":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == "mlp":
        mlp_params = 0
        for name, param in model.named_parameters():
            if "mlp" in name:
                mlp_params += param.numel()
        return mlp_params
    elif mode == "lora":
        lora_params = 0
        for name, param in model.named_parameters():
            if "lora" in name:
                lora_params += param.numel()
        return lora_params
    return 0


def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def verify_config(hidden_size=1024, num_layers=28, mlp_ratio=3.0,
                  use_lora=False, lora_rank=8, lora_alpha=16,
                  lora_mode="full", lora_dropout=0.05, vocab_size=151936, tie_embeddings=True):
    config = Qwen3Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=int(hidden_size * mlp_ratio),
        vocab_size=vocab_size,
        tie_word_embeddings=tie_embeddings,
        use_lora=use_lora,
        lora_mode=lora_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    model = Qwen3ForCausalLM(config)
    
    total_params = calculate_params(model, "total")
    trainable_params = calculate_params(model, "trainable")
    mlp_params = calculate_params(model, "mlp")
    lora_params = calculate_params(model, "lora")
    
    return {
        "config": config,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "mlp_params": mlp_params,
        "lora_params": lora_params
    }


def compare_configs():
    print("=" * 100)
    print("参数量对比验证 (基于 config.json)")
    print("=" * 100)
    
    configs = [
        {"mlp_ratio": 3.0, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Baseline (ratio=3.0, no LoRA)"},
        {"mlp_ratio": 2.0, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Dense (ratio=2.0, no LoRA)"},
        {"mlp_ratio": 1.0, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Dense (ratio=1.0, no LoRA)"},
        {"mlp_ratio": 0.5, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Dense (ratio=0.5, no LoRA)"},
        {"mlp_ratio": 8.0, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Dense (ratio=8.0, no LoRA)"},
        {"mlp_ratio": 4.0, "use_lora": False, "lora_rank": 0, "lora_mode": "none", "desc": "Dense (ratio=4.0, no LoRA)"},
    ]
    
    results = []
    for cfg in configs:
        result = verify_config(
            mlp_ratio=cfg["mlp_ratio"],
            use_lora=cfg["use_lora"],
            lora_rank=cfg["lora_rank"],
            lora_mode=cfg["lora_mode"]
        )
        results.append({**cfg, **result})
    
    print(f"\n{'配置':<35} {'Total':<12} {'Trainable':<12} {'MLP':<12}")
    print("-" * 85)
    
    baseline_params = results[0]["total_params"]
    for r in results:
        ratio = r["total_params"] / baseline_params if baseline_params > 0 else 1.0
        diff = (r["total_params"] - baseline_params) / baseline_params * 100
        
        print(f"{r['desc']:<35} "
              f"{format_number(r['total_params']):<12} "
              f"{format_number(r['trainable_params']):<12} "
              f"{format_number(r['mlp_params']):<12}")
        
        if r != results[0]:
            status = "✓" if abs(diff) < 1.0 else "✗"
            print(f"  → vs Baseline: {diff:+.2f}% {status}")
        print()
    
    print("=" * 100)
    print("MLP 参数量计算公式:")
    print("=" * 100)
    print("Dense MLP: 3 * hidden_size * intermediate_size")
    print("LoRA MLP: 3 * hidden_size * intermediate_size + 3 * rank * (hidden_size + intermediate_size)")
    print()
    print("示例计算 (hidden_size=1024, intermediate_size=3072):")
    print("  Dense: 3 * 1024 * 3072 = 9,437,184 = 9.44M per layer")
    print("  LoRA (rank=8): 9.44M + 3 * 8 * (1024 + 3072) = 9.44M + 98,304 = 9.54M per layer")
    print("=" * 100)


def find_rank_for_params(target_ratio, base_ratio, base_params, hidden_size=1024, num_layers=28, lora_alpha=16):
    intermediate_size = int(hidden_size * target_ratio)

    base_intermediate_size = int(hidden_size * base_ratio)
    base_mlp_params_per_layer = 3 * hidden_size * base_intermediate_size

    lora_params_needed_per_layer = base_mlp_params_per_layer
    rank_per_layer = lora_params_needed_per_layer / (3 * (hidden_size + intermediate_size))

    return int(round(rank_per_layer))


def auto_find_ranks():
    print("=" * 100)
    print("自动计算使参数量恒定的 LoRA Rank (纯 LoRA 模式)")
    print("=" * 100)

    hidden_size = 1024
    num_layers = 28
    base_ratio = 3.0
    base_intermediate_size = int(hidden_size * base_ratio)

    base_config = verify_config(mlp_ratio=base_ratio, use_lora=False)
    base_params = base_config["total_params"]

    print(f"\n基准配置: ratio={base_ratio}, Total Params = {format_number(base_params)}")
    print(f"hidden_size={hidden_size}, num_layers={num_layers}")
    print(f"intermediate_size={base_intermediate_size}")
    print()

    ratios = [0.25, 0.5, 1.0]

    print(f"{'Target Ratio':<15} {'Rank Needed':<15} {'Actual Params':<15} {'Diff %':<10}")
    print("-" * 60)

    for ratio in ratios:
        rank = find_rank_for_params(ratio, base_ratio, base_params, hidden_size, num_layers)

        result = verify_config(
            mlp_ratio=ratio,
            use_lora=True,
            lora_rank=rank,
            lora_mode="lora_only"
        )

        diff = (result["total_params"] - base_params) / base_params * 100

        print(f"{ratio:<15} {rank:<15} {format_number(result['total_params']):<15} {diff:+.2f}%")

    print("=" * 100)


def compare_lora_experiments():
    print("=" * 100)
    print("实验3: LoRA 参数量恒定实验 (L=28, d=[0.25d, 0.5d, 1d, 2d, 4d, 8d])")
    print("=" * 100)

    hidden_size = 1024
    num_layers = 28
    base_ratio = 3.0

    base_config = verify_config(mlp_ratio=base_ratio, use_lora=False)
    base_params = base_config["total_params"]

    print(f"\nBaseline (无 LoRA): ratio={base_ratio}, Total Params = {format_number(base_params)}")
    print(f"hidden_size={hidden_size}, num_layers={num_layers}")
    print()

    ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"\n{'Ratio':<8} {'Rank':<10} {'Total':<12} {'Trainable':<12} {'Diff %':<10}")
    print("-" * 60)

    results = []
    for ratio in ratios:
        rank = find_rank_for_params(ratio, base_ratio, base_params, hidden_size, num_layers)

        if rank <= 0:
            print(f"{ratio:<8} {'N/A':<10} {'N/A (已超baseline)':<30} {'N/A':<12}")
            continue

        result = verify_config(
            mlp_ratio=ratio,
            use_lora=True,
            lora_rank=rank,
            lora_mode="lora_only"
        )

        diff = (result["total_params"] - base_params) / base_params * 100

        results.append({
            "ratio": ratio,
            "mode": "lora_only",
            "rank": rank,
            "total": result["total_params"],
            "trainable": result["trainable_params"],
            "diff": diff
        })

        print(f"{ratio:<8} {rank:<10} "
              f"{format_number(result['total_params']):<12} "
              f"{format_number(result['trainable_params']):<12} "
              f"{diff:+.2f}%")

    print("\n" + "=" * 100)
    print("实验配置汇总")
    print("=" * 100)
    print("\n实验 1: 更改 MLP 中间层维度，保持层数不变 (L=28)")
    for ratio in ratios:
        result = verify_config(mlp_ratio=ratio, use_lora=False)
        diff = (result["total_params"] - base_params) / base_params * 100
        print(f"  ratio={ratio:<5} -> Total: {format_number(result['total_params']):<12} (diff: {diff:+.2f}%)")

    print("\n实验 2: 更改 MLP 中间层维度，通过更改层数保持参数量恒定")
    print("  (已在 train_qwen3.py 中通过 --iso_parameter 参数实现)")

    print("\n实验 3: 更改 MLP 中间层维度，通过 LoRA 保持参数量恒定 (L=28)")
    for r in results:
        print(f"  ratio={r['ratio']:<5} + LoRA(lora_only, rank={r['rank']}) -> "
              f"Total: {format_number(r['total']):<12}, Trainable: {format_number(r['trainable']):<12}")

    print("=" * 100)


if __name__ == "__main__":
    compare_configs()
    print()
    auto_find_ranks()
    print()
    compare_lora_experiments()

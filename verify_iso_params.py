import torch
from src.model.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from src.model.qwen3.configuration_qwen3 import Qwen3Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def verify():
    base_k = 3.0
    base_L = 28
    hidden_size = 1024
    # 经验公式分值: L * (6 + 3k)
    # 注意：Attention = 6d^2 (q:2, k:1, v:1, o:2), MLP = 3kd^2
    base_score = base_L * (6 + 3 * base_k)
    
    ratios = [0.5, 1.0, 3.0, 8.0]
    
    print(f"{'Ratio (k)':<10} | {'Layers (L)':<10} | {'Params (M)':<15} | {'Diff (%)':<10}")
    print("-" * 55)
    
    # 获取基准参数量 (L=28, k=3.0)
    config_base = Qwen3Config.from_pretrained("/data/250010109/qwen_exp/configs")
    model_base = Qwen3ForCausalLM(config_base)
    base_params = count_parameters(model_base)
    
    for k in ratios:
        # 计算维持参数量所需的层数
        L = int(round(base_score / (6 + 3 * k)))
        config = Qwen3Config.from_pretrained("/data/250010109/qwen_exp/configs")
        config.num_hidden_layers = L
        config.intermediate_size = int(config.hidden_size * k)
        # Manually update layer_types to match new num_hidden_layers
        config.layer_types = ["full_attention"] * L 
        model = Qwen3ForCausalLM(config)
        params = count_parameters(model)
        diff = (params - base_params) / base_params * 100
        print(f"{k:<10} | {L:<10} | {params/1e6:<15.2f} | {diff:<10.2f}%")

if __name__ == "__main__":
    verify()

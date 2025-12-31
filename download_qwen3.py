import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "Qwen/Qwen3-0.6B"
config_dir = "/data/250010109/qwen_exp/configs"
model_dir = "/data/250010109/qwen_exp/model"

os.makedirs(config_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading configs for {repo_id}...")
# Download only config related files
snapshot_download(
    repo_id=repo_id,
    local_dir=config_dir,
    allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
)

print(f"Downloading model scripts for {repo_id}...")
# Download only modeling scripts
snapshot_download(
    repo_id=repo_id,
    local_dir=model_dir,
    allow_patterns=["modeling_*.py", "configuration_*.py", "tokenization_*.py", "qwen_*.py"]
)

print("Download complete.")

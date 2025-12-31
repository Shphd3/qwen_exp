import os

# Set environment variables immediately after import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"

import time
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# Import from src
from src.data_utils import BinaryDataset
# We will use the local modeling code
from src.model.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from src.model.qwen3.configuration_qwen3 import Qwen3Config

def chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int, ignore_index: int) -> torch.Tensor:
    logits_2d = logits.reshape(-1, logits.size(-1))
    labels_1d = labels.reshape(-1)

    if chunk_size <= 0:
        return F.cross_entropy(logits_2d.float(), labels_1d, ignore_index=ignore_index)

    total_loss = logits_2d.new_zeros((), dtype=torch.float32)
    total_valid_tokens = labels_1d.new_zeros((), dtype=torch.long)
    num_tokens = labels_1d.numel()

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        logits_chunk = logits_2d[start:end].float()
        labels_chunk = labels_1d[start:end]

        loss_chunk = F.cross_entropy(
            logits_chunk,
            labels_chunk,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = total_loss + loss_chunk
        total_valid_tokens = total_valid_tokens + labels_chunk.ne(ignore_index).sum()

    return total_loss / total_valid_tokens.clamp_min(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for Qwen3-0.6B")

    # Batch & training config
    parser.add_argument("--local_batch_size", type=int, default=12)
    parser.add_argument("--global_batch_size", type=int, default=768)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")
    
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--config_dir", type=str, default="/data/250010109/qwen_exp/configs")
    parser.add_argument("--data_dir", type=str, default="/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10")
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="/data/share/109_cache_dir/qwen_exp/output")
    parser.add_argument("--model_name", type=str, default="qwen3-0.6b-scratch")

    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--total_tokens", type=float, default=10e9, help="Total tokens to train on (default 10B)")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--loss_chunk_size", type=int, default=2048)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--zero_stage", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--mlp_ratio", type=float, default=None, help="Ratio of intermediate_size to hidden_size")
    parser.add_argument("--iso_parameter", action="store_true", help="Auto-adjust layers to maintain total parameter count")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) for MLP layers")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_mode", type=str, default="lora", choices=["lora", "lora_bias"],
                        help="LoRA mode: 'lora' (only LoRA, no base), 'lora_bias' (frozen base + LoRA)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for low-rank adaptation")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")

    # DeepSpeed
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    return args

def get_ds_config(args):
    ds_config = {
        "train_batch_size": args.global_batch_size,
        "train_micro_batch_size_per_gpu": args.local_batch_size,
        "steps_per_print": 10,
        "prescale_gradients": False,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": args.use_bf16
        },
        "zero_optimization": {
            "stage": args.zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
    }

    if args.zero_stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })

    return ds_config

def main():
    args = parse_args()
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Prepare Output Dirs
    full_output_dir = os.path.join(args.output_dir, args.model_name)
    ckpt_dir = os.path.join(full_output_dir, "checkpoints")
    log_dir = os.path.join(full_output_dir, "logs")
    
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Output directory: {full_output_dir}")

    # TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to {log_dir}")

    # Model
    if rank == 0:
        print("Initializing Model from scratch...", flush=True)
    config = Qwen3Config.from_pretrained(args.config_dir)

    # ISO-Parameter Logic (Optional, kept for flexibility but default off)
    if args.iso_parameter and args.mlp_ratio is not None:
        # Base configuration: k=3.0, L=28
        # Parameter score proportional to L * (6 + 3k)
        # Attention = 6d^2 (q:2, k:1, v:1, o:2), MLP = 3kd^2
        base_k = 3.0
        base_L = 28
        base_score = base_L * (6 + 3 * base_k)
        
        new_k = args.mlp_ratio
        # New L = base_score / (6 + 3 * new_k)
        new_L = int(round(base_score / (6 + 3 * new_k)))
        
        if rank == 0:
            print(f"ISO-Parameter Mode: Adjusting num_hidden_layers from {config.num_hidden_layers} to {new_L} "
                  f"to match MLP ratio {new_k} (base ratio {base_k}, base layers {base_L})")
        
        config.num_hidden_layers = new_L
        config.layer_types = ["full_attention"] * new_L

    if args.mlp_ratio is not None:
        config.intermediate_size = int(config.hidden_size * args.mlp_ratio)
        if rank == 0:
            print(f"Setting intermediate_size to {config.intermediate_size} (ratio {args.mlp_ratio})")
    
    # LoRA Configuration
    if args.use_lora:
        config.use_lora = True
        config.lora_mode = args.lora_mode
        config.lora_rank = args.lora_rank
        config.lora_alpha = args.lora_alpha
        config.lora_dropout = args.lora_dropout
        if rank == 0:
            print(f"LoRA enabled: mode={args.lora_mode}, rank={args.lora_rank}, alpha={args.lora_alpha}")
    else:
        config.use_lora = False
        config.lora_mode = "none"
            
    model = Qwen3ForCausalLM(config)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("Gradient Checkpointing enabled.")

    # Data
    if rank == 0:
        print(f"Loading Binary Dataset from {args.data_dir}...", flush=True)
    
    dataset = BinaryDataset(args.data_dir, args.seq_len, dtype=np.uint32)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=args.data_shuffle)
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # Optimizer & Scheduler
    ds_config = get_ds_config(args)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Calculate steps for scheduler
    tokens_per_global_step = args.global_batch_size * args.seq_len
    total_training_tokens = args.total_tokens
    dataset_tokens = len(dataset) * args.seq_len
    total_tokens_to_train = min(total_training_tokens, dataset_tokens * args.epochs)
    total_steps = int(total_tokens_to_train // tokens_per_global_step)
    
    if rank == 0:
        print(f"Total training tokens: {total_tokens_to_train}")
        print(f"Tokens per global step: {tokens_per_global_step}")
        print(f"Total global steps for scheduler: {total_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps 
    )
    
    # Initialize DeepSpeed Engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config
    )

    # Resuming logic
    global_step = 0 
    total_tokens = 0
    samples_seen = 0 
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(ckpt_dir)
        if load_path:
            if rank == 0:
                print(f"Resuming from checkpoint: {load_path}", flush=True)
            if client_state:
                global_step = client_state.get('global_step', 0)
                total_tokens = client_state.get('total_tokens', 0)
                samples_seen = client_state.get('samples_seen', 0)
                if rank == 0:
                    print(f"Restored global_step: {global_step}, total_tokens: {total_tokens}, samples_seen: {samples_seen}")
        else:
            if rank == 0:
                print(f"Warning: --resume specified but no checkpoint found. Starting from scratch.", flush=True)
    
    # Training Loop
    model_engine.train()
    accum_loss = 0.0
    total_local_batch = args.local_batch_size * world_size
    gradient_accumulation_steps = args.global_batch_size // total_local_batch

    if rank == 0:
        pbar = tqdm(desc="Training", total=total_steps)
        print(f"Training Config:")
        print(f"  GPUs: {world_size}")
        print(f"  Local Batch Size per GPU: {args.local_batch_size}")
        print(f"  Total Local Batch Size: {total_local_batch}")
        print(f"  Global Batch Size: {args.global_batch_size}")
        print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"  Total Dataset Samples: {len(dataset)}")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, (source, target, real_lens) in enumerate(dataloader):
            # Resume skip logic
            if args.resume and samples_seen > 0:
                current_total_samples = epoch * len(dataset) + step * total_local_batch
                if current_total_samples < samples_seen:
                    continue

            start_time = time.time()
            source = source.to(model_engine.device)
            target = target.to(model_engine.device)
            
            outputs = model_engine(source, output_hidden_states=False)
            loss = chunked_cross_entropy(outputs.logits, target, chunk_size=args.loss_chunk_size, ignore_index=151643) # Using default pad_token_id
            
            model_engine.backward(loss)
            model_engine.step()
            
            batch_time = time.time() - start_time
            
            # Token and sample counting
            current_tokens = torch.tensor(real_lens.sum().item(), device=model_engine.device)
            dist.all_reduce(current_tokens, op=dist.ReduceOp.SUM)
            total_tokens += current_tokens.item()
            samples_seen += total_local_batch

            # Logging
            if rank == 0:
                accum_loss += loss.item()
                opt_step = model_engine.global_steps
                lr = model_engine.get_lr()[0] if model_engine.get_lr() else 0.0
                is_boundary = model_engine.is_gradient_accumulation_boundary()
                
                if step % 10 == 0:
                    log_msg = f"batch: {opt_step}-{step+1}, loss: {loss.item():.6f}, lr: {lr:.6e}, opt_step: {opt_step}, batch_time: {batch_time:.3f}, total_tokens: {total_tokens}"
                    print(log_msg, flush=True)
                
                if is_boundary:
                    avg_loss = accum_loss / gradient_accumulation_steps
                    writer.add_scalar('Loss/train', avg_loss, opt_step)
                    writer.add_scalar('Tokens/total', total_tokens, opt_step)
                    writer.add_scalar('LR', lr, opt_step)
                    accum_loss = 0.0 
                    pbar.update(1)
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.2e}", 'opt_step': opt_step})
            
            # Save Checkpoint
            opt_step = model_engine.global_steps
            if opt_step > 0 and opt_step % args.save_interval == 0 and model_engine.is_gradient_accumulation_boundary():
                client_state = {
                    'global_step': global_step,
                    'total_tokens': total_tokens,
                    'samples_seen': samples_seen
                }
                model_engine.save_checkpoint(ckpt_dir, f"checkpoint-{opt_step}", client_state=client_state)

            # Stopping condition
            if total_tokens >= total_training_tokens:
                if rank == 0:
                    print(f"Reached {total_tokens} tokens (>= {total_training_tokens}), stopping training.")
                break
            
            global_step += 1
        
        if total_tokens >= total_training_tokens:
            break

    if rank == 0:
        print("Training finished.")
        writer.close()
        pbar.close()

if __name__ == "__main__":
    main()

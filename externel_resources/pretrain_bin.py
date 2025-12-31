import os
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
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# Import from src
from src.utils.data_utils import TokenizedJSONLData, BinaryDataset
from src.model.myqwen import MyQwen3ForCausalLM


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
    parser = argparse.ArgumentParser(description="Training configuration")

    # Batch & training config
    parser.add_argument("--local_batch_size", type=int, default=8)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")
    
    # Model config
    parser.add_argument("--moe_type", type=str, default="none") # ["none", "moe", "multihead"]
    parser.add_argument("--moe_intermediate_size", type=int, default=1536)
    parser.add_argument("--moe_head_num", type=int, default=8)
    parser.add_argument("--expert_per_token", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--gating_reference", type=str, choices=["oracle", "switch"], default="switch")
    parser.add_argument("--moe_implementation", type=str, choices=["gated", "cat"], default="gated")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", type=str, default="./cache_dir/Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_type", type=str, default="jsonl", choices=["jsonl", "bin"])
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_name", type=str, default="model")

    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in ckpt_dir")

    parser.add_argument("--total_tokens", type=float, default=50e9, help="Total number of tokens to train on")

    parser.add_argument("--debug_data", action="store_true", default=False)

    parser.add_argument("--loss_chunk_size", type=int, default=2048)
    
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    # ZeRO Stage
    parser.add_argument("--zero_stage", type=int, default=1, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO optimization stage")

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
        }
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

def prepare_model(args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    config.moe_type = args.moe_type
    config.moe_intermediate_size = args.moe_intermediate_size
    config.moe_head_num = args.moe_head_num
    config.num_experts_per_tok = args.expert_per_token
    config.num_experts = args.num_experts
    config.gating_reference = args.gating_reference
    config.moe_implementation = args.moe_implementation
    config.norm_topk_prob = True

    model = MyQwen3ForCausalLM(config)
    return model

def main():
    args = parse_args()
    
    # 初始化deepspeed
    deepspeed.init_distributed()
    
    # Prepare Output Dirs
    if args.ckpt_dir and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    
    full_output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir, exist_ok=True)

    # TensorBoard
    writer = None
    if deepspeed.comm.get_rank() == 0:
        writer = SummaryWriter(log_dir=full_output_dir)
        print(f"Logging to {full_output_dir}")

    # Model
    print("Initializing Model...", flush=True)
    model = prepare_model(args)
    if args.gradient_checkpointing:
        print("Enabling Gradient Checkpointing...", flush=True)
        model.gradient_checkpointing_enable()
    print("Model Initialized.", flush=True)
    
    # Get pad_token_id from model config
    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        print("Warning: pad_token_id is None in config. Using default 151643.")
        pad_token_id = 151643

    # Data
    print(f"Initializing Dataset (Type: {args.dataset_type})...", flush=True)
    sampler = None
    total_samples = 0
    if args.dataset_type == "bin":
        # Check for dataset_info.json to infer dtype
        info_path = os.path.join(args.data_dir, "dataset_info.json")
        dtype = np.uint32
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                meta = json.load(f)
                if meta.get("dtype") == "uint16":
                    dtype = np.uint16
        
        dataset = BinaryDataset(args.data_dir, args.seq_len, dtype=dtype)
        sampler = DistributedSampler(dataset, shuffle=args.data_shuffle)
        total_samples = len(dataset)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
        dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer, shuffle=args.data_shuffle)
        # IterableDataset 没有 len，这里设为一个极大值或 0
        total_samples = 0 
    
    print("Dataset Initialized.", flush=True)
    
    # DeepSpeed Engine
    ds_config = get_ds_config(args)
    
    print("Initializing DeepSpeed Engine...", flush=True)
    
    # Optimizer & Scheduler (Matched with external_resources/scripts/pretrain_qwen.py)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    # 在 main() 中 deepspeed.initialize 之前

    # Correct calculation of gradient accumulation steps (inferred by DS, but useful for logging)
    world_size = deepspeed.comm.get_world_size()
    total_local_batch = args.local_batch_size * world_size
    
    if args.global_batch_size % total_local_batch != 0:
        raise ValueError(f"Global batch size ({args.global_batch_size}) must be divisible by total local batch size ({total_local_batch} = {args.local_batch_size} * {world_size} GPUs)")
    
    # 根据全局批次大小，计算梯度累加步数
    # 全局批次大小 = 本地批次大小 * 世界大小 * 梯度累加步数
    gradient_accumulation_steps = args.global_batch_size // total_local_batch
    
    # 动态计算总步数
    # 这里的目标是训练 total_tokens，我们需要将其转换为优化器步数（global steps）
    # 每一步处理的 token 数 = global_batch_size * seq_len
    tokens_per_global_step = args.global_batch_size * args.seq_len
    total_training_tokens = args.total_tokens
    
    # 如果数据集的总 token 数 * epochs 小于 total_training_tokens，则以数据集为准
    if args.dataset_type == "bin":
        dataset_tokens = len(dataset) * args.seq_len
    else:
        # IterableDataset 估计一个无限大的值，完全由 args.total_tokens 控制
        dataset_tokens = float('inf')
        
    total_tokens_to_train = min(total_training_tokens, dataset_tokens * args.epochs)
    total_steps = int(total_tokens_to_train // tokens_per_global_step)
    
    if deepspeed.comm.get_rank() == 0:
        print(f"Total training tokens: {total_tokens_to_train}")
        print(f"Tokens per global step: {tokens_per_global_step}")
        print(f"Total global steps for scheduler: {total_steps}")

    # 使用动态步数
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=2000, 
        num_training_steps=total_steps 
    )
    
    # 初始化Deepspeed 引擎！
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config
    )

    # 尝试加载检查点
    global_step = 0 # 这里的 global_step 记录的是 micro-batch 数量
    total_tokens = 0
    samples_seen = 0 # 新增：记录总共见过的样本数，用于跨卡数 resume
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.ckpt_dir)
        if load_path:
            print(f"Resuming from checkpoint: {load_path}", flush=True)
            if client_state:
                global_step = client_state.get('global_step', 0)
                total_tokens = client_state.get('total_tokens', 0)
                samples_seen = client_state.get('samples_seen', 0)
                print(f"Restored global_step: {global_step}, total_tokens: {total_tokens}, samples_seen: {samples_seen}")
        else:
            print(f"Warning: --resume specified but no checkpoint found in {args.ckpt_dir}. Starting from scratch.", flush=True)
    else:
        print("Starting training from scratch (no --resume specified).", flush=True)
    
    print("DeepSpeed Engine Initialized.", flush=True)

    # DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.local_batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False if sampler else None # Sampler handles shuffle
    )

    # Training Loop
    # global_step and total_tokens are initialized/restored above
    accum_loss = 0.0
    
    if deepspeed.comm.get_rank() == 0:
        pbar = tqdm(desc="Training")
        # Ensure log file exists and is writable
        log_file_path = os.path.join(full_output_dir, "training_debug.log")
        print(f"Debug logs will be written to {log_file_path}")
    else:
        log_file_path = None
    
    
    if deepspeed.comm.get_rank() == 0:
        print(f"Training Config:")
        print(f"  GPUs: {world_size}")
        print(f"  Local Batch Size per GPU: {args.local_batch_size}")
        print(f"  Total Local Batch Size: {total_local_batch}")
        print(f"  Global Batch Size: {args.global_batch_size}")
        print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
        if args.dataset_type == "bin":
             total_samples = len(dataset)
             total_training_tokens = total_samples * args.seq_len * args.epochs
             print(f"  Total Dataset Samples: {total_samples}")
             print(f"  Total Training Tokens (Approx): {total_training_tokens} (Samples * SeqLen * Epochs)")
             print(f"  Note: With {world_size} GPUs, the dataset is sharded. Each GPU sees ~{total_samples // world_size} samples per epoch.")


    for epoch in range(args.epochs):
        # Set epoch for sampler
        if sampler:
            sampler.set_epoch(epoch)
            
        for step, (source, target, real_lens) in enumerate(dataloader):
            # 优化后的断点续训跳过逻辑：
             # 基于 samples_seen 来决定当前进程是否需要跳过当前的 micro-batch。
             # 这种方式比依赖 len(dataloader) 更健壮，因为它在不同卡数下更具参考价值。
            if args.resume and samples_seen > 0 and total_samples > 0:
                # 计算当前进程在这个 epoch 已经处理过的样本数
                # 这是一个近似判断，旨在确保恢复到大致正确的进度
                current_total_samples = epoch * total_samples + step * args.local_batch_size * world_size
                if current_total_samples < samples_seen:
                    if step % 100 == 0 and deepspeed.comm.get_rank() == 0:
                        print(f"Skipping step {step} (already processed {current_total_samples} samples)...", flush=True)
                    continue

            start_time = time.time()

            if args.debug_data and epoch == 0 and step == 0:
                rank = deepspeed.comm.get_rank()
                s0 = int(source[0, 0].item())
                s1 = int(source[0, 1].item())
                ssum = int(source[0, :16].sum().item())
                tsum = int(target[0, :16].sum().item())
                pad_id = pad_token_id
                pad_cnt = int((target[0] == pad_id).sum().item())
                valid_cnt = int((target[0] != pad_id).sum().item())
                dbg = torch.tensor([rank, s0, s1, ssum, tsum, pad_cnt, valid_cnt], device=model_engine.device, dtype=torch.long)
                gathered = [torch.zeros_like(dbg) for _ in range(world_size)]
                dist.all_gather(gathered, dbg)
                if rank == 0:
                    gathered_rows = [g.tolist() for g in gathered]
                    print(f"debug_data: {gathered_rows}", flush=True)

            source = source.to(model_engine.device)
            target = target.to(model_engine.device)
            
            outputs = model_engine(source, output_hidden_states=False)

            loss = chunked_cross_entropy(outputs.logits, target, chunk_size=args.loss_chunk_size, ignore_index=pad_token_id)
            
            # Backward
            model_engine.backward(loss)
            model_engine.step()
            
            batch_time = time.time() - start_time
            
            # Accurate token and sample counting
            current_tokens = torch.tensor(real_lens.sum().item(), device=model_engine.device)
            dist.all_reduce(current_tokens, op=dist.ReduceOp.SUM)
            total_tokens += current_tokens.item()
            samples_seen += args.local_batch_size * world_size

            # Logging
            if deepspeed.comm.get_rank() == 0:
                accum_loss += loss.item()
                local_batch_idx = step + 1
                # 获取 DeepSpeed 内部的全局优化步数
                opt_step = model_engine.global_steps
                lr = model_engine.get_lr()[0] if model_engine.get_lr() else 0.0
                is_boundary = model_engine.is_gradient_accumulation_boundary() if hasattr(model_engine, "is_gradient_accumulation_boundary") else (local_batch_idx % gradient_accumulation_steps == 0)
                
                log_msg = f"batch: {opt_step}-{local_batch_idx}, loss: {loss.item():.6f}, lr: {lr:.6e}, opt_step: {opt_step}, accum_boundary: {int(is_boundary)}, batch_time: {batch_time:.3f}, total_tokens: {total_tokens}"
                print(log_msg, flush=True)
                
                # Only log to TensorBoard at the end of a gradient accumulation step
                if is_boundary:
                    avg_loss = accum_loss / gradient_accumulation_steps
                    writer.add_scalar('Loss/train', avg_loss, opt_step)
                    writer.add_scalar('Tokens/total', total_tokens, opt_step)
                    writer.add_scalar('LR', lr, opt_step)
                    writer.add_scalar('WeightDecay', args.weight_decay, opt_step)
                    accum_loss = 0.0 # Reset accumulated loss
                
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss.item():.6f}", 'lr': f"{lr:.2e}", 'opt_step': opt_step})

            global_step += 1

            # 修改保存逻辑：改用全局优化步数 (opt_step) 作为触发条件
            # 这样无论卡数如何，保存的物理频率是一致的
            opt_step = model_engine.global_steps
            if opt_step > 0 and opt_step % args.save_interval == 0 and model_engine.is_gradient_accumulation_boundary():
                client_state = {
                    'global_step': global_step,
                    'total_tokens': total_tokens,
                    'samples_seen': samples_seen
                }
                model_engine.save_checkpoint(args.ckpt_dir, f"checkpoint-{opt_step}", client_state=client_state)

            # Stop training if total tokens exceed total_training_tokens
            if total_tokens >= total_training_tokens:
                if deepspeed.comm.get_rank() == 0:
                    print(f"Reached {total_tokens} tokens (>= {total_training_tokens}), stopping training.")
                break
        
        if total_tokens >= total_training_tokens:
            break

    if deepspeed.comm.get_rank() == 0:
        writer.close()
        pbar.close()

if __name__ == "__main__":
    main()

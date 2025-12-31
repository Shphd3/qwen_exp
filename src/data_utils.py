import os
import json
import random
import torch
import numpy as np
import glob
import bisect # 用于高效查找文件索引
from torch.utils.data import IterableDataset, Dataset, get_worker_info
import torch.distributed as dist

class BinaryDataset(Dataset):
    def __init__(self, dataset_dir, max_seq_len, dtype=np.uint32):
        self.dataset_dir = dataset_dir
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.item_size = np.dtype(dtype).itemsize # 每个token占用的字节数
        
        # 递归查找所有 .bin 文件
        self.bin_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.bin"), recursive=True))
        
        if not self.bin_files:
            print(f"Warning: No .bin files found in {dataset_dir}")
        
        self.file_offsets = [] # 存储 (全局起始 token 索引, 全局结束 token 索引, 文件路径)
        self.file_starts = [] # 存储每个文件对应的全局起始 token 索引，用于 bisect 查找 
        self.cumulative_tokens = 0 # 累计的总 token 数
        
        # 遍历所有 .bin 文件，计算它们的 token 范围和全局偏移
        for f in self.bin_files:
            try:
                size_bytes = os.path.getsize(f)
                num_tokens = size_bytes // self.item_size
                
                if num_tokens == 0:
                    print(f"Warning: Skipping empty file {f}") # 处理空文档！
                    continue
                    
                self.file_starts.append(self.cumulative_tokens) # 记录当前文件的全局起始 token 索引
                self.file_offsets.append((self.cumulative_tokens, self.cumulative_tokens + num_tokens, f))
                self.cumulative_tokens += num_tokens
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                continue
        
        # 计算总样本数。每个样本是 max_seq_len + 1 个 token (max_seq_len 用于 source, 1 用于 target 的第一个 token)
        # 实际上，这里是 (total_tokens - 1) // max_seq_len，因为 target 比 source 错开一位
        # 这里的 (max_seq_len + 1) 可能是为了确保能取到 max_seq_len 长度的 source 和 target
        self.total_samples = max(0, self.cumulative_tokens // (max_seq_len + 1))
        # print(f"Loaded {len(self.bin_files)} bin files. Total tokens: {self.cumulative_tokens}. Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        根据索引获取单个样本。
        一个样本包含 max_seq_len 个 token 作为输入 (x) 和 max_seq_len 个 token 作为目标 (y)。
        因此，需要从数据中读取 max_seq_len + 1 个 token。
        """
        #  计算当前样本在整个数据集中的起始 token 索引
        start_token_idx = idx * (self.max_seq_len + 1)
        
        # 使用 bisect 查找 start_token_idx 所在的文件的索引
        # bisect_right 返回插入点，减 1 得到实际文件索引
        file_idx = bisect.bisect_right(self.file_starts, start_token_idx) - 1
        start_global, end_global, filepath = self.file_offsets[file_idx]
        # 计算当前样本在当前文件中的起始偏移量
        offset_in_file = start_token_idx - start_global
        tokens_to_read = self.max_seq_len + 1 # 需要读取的 token 数量 (max_seq_len 用于 x, 1 用于 y 的第一个 token)
        
        data = [] # 存储读取到的 token 块
        current_idx = start_token_idx # 当前全局读取位置
        remaining = tokens_to_read # 还需要读取的 token 数量
        curr_file_idx = file_idx # 当前正在处理的文件索引

        # 循环读取，处理跨文件的情况       
        while remaining > 0 and curr_file_idx < len(self.file_offsets):
            f_start, f_end, f_path = self.file_offsets[curr_file_idx]
            
            # 计算在当前文件中的本地起始和结束索引
            local_start = current_idx - f_start
            local_end = min(f_end - f_start, local_start + remaining) # 确保不超过文件边界
            read_len = local_end - local_start  # 实际从当前文件读取的长度
            
            # Read using memmap (mapping full file to avoid alignment issues, OS handles paging)
            # 使用内存映射读取数据块
            mm = np.memmap(f_path, dtype=self.dtype, mode='r')
            chunk = mm[local_start:local_start+read_len]
            
            # 将 numpy 数组转换为 int64 类型的 PyTorch Tensor，以兼容 PyTorch 的 embedding 层
            data.append(torch.from_numpy(chunk.astype(np.int64)))
            
            remaining -= read_len
            current_idx += read_len
            curr_file_idx += 1
            
        if remaining > 0:
            # 如果读取到所有文件末尾仍不足 tokens_to_read，则进行填充
            padding = torch.zeros(remaining, dtype=torch.long)
            data.append(padding)
            
        cat_data = torch.cat(data) # 将所有读取到的 token 块拼接起来
        
        x = cat_data[:-1] # 输入序列 (source)
        y = cat_data[1:] # 目标序列 (target)
        
        return x, y, len(x)

class TokenizedJSONLData(IterableDataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding=True, shuffle=True, buffer_size=1000) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.shuffle = shuffle
        self.buffer_size = buffer_size # 用于 shuffle 的缓冲区大小
        
        # 递归收集所有子文件夹中的 .jsonl 或 .json 文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.jsonl') or filename.endswith('.json'):
                    self.files.append(os.path.join(root, filename))
        self.files = sorted(self.files)
        
        if not self.files:
            print(f"Warning: No .jsonl or .json files found in {dataset_dir}")
            
    def _parse_line(self, line):
        """
        解析 JSONL 行，提取 'text' 字段。
        """
        try:
            line = line.strip()
            if not line:
                return None
            data = json.loads(line)
            return data.get('text', None)
        except json.JSONDecodeError:
            return None

    def _tokenize(self, text):
        """
        对文本进行分词，并生成输入 (src) 和目标 (tgt) 序列。
        """
        if self.padding:
            enc = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            )
            token_ids = enc.input_ids
            attn = enc.attention_mask
            # 获取 attention mask
            # real_len 通过 attention mask 计算，更准确地反映实际 token 长度
            real_len = int(attn[0, 1:].sum().item())
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = int(token_ids.size(1) - 1)
        
        # Input: :-1, Target: 1:
        # Returns: source, target, length
        return token_ids[0, :-1], token_ids[0, 1:], real_len

    def __iter__(self):
        """
        IterableDataset 的核心方法，返回一个迭代器。
        """
        worker_info = get_worker_info()
        
        # 确定当前进程的 rank 和 world_size (用于分布式训练)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # 确定当前 worker 的 id 和总 worker 数量 (用于多进程数据加载)
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
        # 数据分片 (Sharding):
        # 1. 全局分片 (进程级别): 每个进程处理一部分文件
        my_files = self.files[rank::world_size]
        
        # 2. 局部分片 (worker 级别): 每个 worker 处理进程所分配文件的一部分
        my_files = my_files[worker_id::num_workers]
        
        if self.shuffle:
            random.shuffle(my_files)
            
        buffer = []
        
        for filepath in my_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = self._parse_line(line) # 解析行，提取文本
                        if not text:
                            continue
                            
                        # Tokenize
                        src, tgt, length = self._tokenize(text)
                        
                        if self.shuffle:
                            buffer.append((src, tgt, length))
                            if len(buffer) >= self.buffer_size:
                                # 当缓冲区满时，随机取出一个样本 yield，并将最后一个样本移到取出位置
                                idx = random.randint(0, len(buffer) - 1)
                                yield buffer[idx]
                                buffer[idx] = buffer[-1]
                                buffer.pop()
                        else:
                            yield src, tgt, length
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
        
        # Yield remaining buffer
        if self.shuffle and buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item

    def __len__(self):
        # 对于Iterable Dataset, __len__一般不准确，因为其不知道总共有多少样本
        return 100000000

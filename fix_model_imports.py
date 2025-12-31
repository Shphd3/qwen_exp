import re
import os

def fix_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace from ... with from transformers.
    content = re.sub(r'from \.\.\.([\w\.]+) import', r'from transformers.\1 import', content)
    # Replace from .. with from transformers.
    content = re.sub(r'from \.\.([\w\.]+) import', r'from transformers.\1 import', content)
    
    # Special case for .configuration_qwen3
    content = re.sub(r'from \.configuration_qwen3', r'from src.model.qwen3.configuration_qwen3', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

fix_imports('/data/250010109/qwen_exp/src/model/qwen3/modeling_qwen3.py')
fix_imports('/data/250010109/qwen_exp/src/model/qwen3/configuration_qwen3.py')

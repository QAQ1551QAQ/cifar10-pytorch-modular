# src/utils/common.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    固定 Python、NumPy、PyTorch 以及 CUDA 的随机种子，
    并开启 cuDNN 确定性算法，确保在相同硬件下每次运行的结果完全一致（可复现）。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 禁用非确定性优化，牺牲微小性能换取绝对的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"--> [Reproducibility] 随机种子已全局固定为: {seed}")

def get_device() -> torch.device:
    """自动适配计算设备：优先 NVIDIA CUDA，其次 Mac MPS，最后兜底 CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--> [Device] 当前计算设备: {device}")
    return device
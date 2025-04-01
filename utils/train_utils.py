import os
import random
import numpy as np

import torch
from torch.distributed import init_process_group, is_initialized


def seed_everything(seed):
    # determine rank (default to 0 for non-DDP scripts)
    rank = int(os.environ.get("RANK", 0)) if "RANK" in os.environ else 0
    global_seed = seed + rank  # adjust seed per process for DDP
    
    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True # forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.benchmark = False # disables autotuner that selects fastest algo; needed for deterministic behavior


def setup_distributed(seed=42):
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp and not is_initialized():
        init_process_group(backend="nccl", init_method="env://")

    print(f"[{os.environ.get('RANK', 0)}] Setting random seed to {seed}")
    seed_everything(seed)
"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import os
import glob
import wandb
import random
import numpy as np
from accelerate import Accelerator

import torch
from torch.distributed import init_process_group, is_initialized


def seed_everything(seed):
    # determine rank (default to 0 for non-ddp scripts)
    rank = int(os.environ.get("RANK", 0)) if "RANK" in os.environ else 0
    global_seed = seed + rank  # adjust seed per process for ddp

    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True  # deterministic kernels
    torch.backends.cudnn.benchmark = False     # disable autotuner


def setup_distributed(seed=97):
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp and not is_initialized():
        init_process_group(backend="nccl", init_method="env://")

    print(f"[{os.environ.get('RANK', 0)}] Setting random seed to {seed}")
    seed_everything(seed)


def init_wandb(config):
    accelerator = Accelerator()
    if accelerator.is_local_main_process:  # Initialize wandb only on the main process
        # run_name = os.path.basename(config_file).split('.')[0]
        wandb.init(project="DynaDUSt3R", config=config) # , name=run_name)
        # for f in log_files():
        #     print(f"Logging file: {f}")
        #     wandb.save(f)
    else:
        # Prevent other processes from logging to wandb
        os.environ["WANDB_MODE"] = "disabled"


from hydra.core.hydra_config import HydraConfig

# ----- keep top-k and encode tracked metric in filename -----
import re
from omegaconf import OmegaConf

_METRIC_NAME_RE = re.compile(
    r"best_(?P<metric>[A-Za-z0-9_\-]+)_(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_iter_(?P<it>\d+)_epoch_(?P<ep>\d+)\.pth"
)

def _parse_metric_from_name(path):
    m = _METRIC_NAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    try:
        return m.group("metric"), float(m.group("val"))
    except Exception:
        return None, None

def _list_checkpoints(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "best_*.pth")))

def save_best_model(accelerator, model, optimizer, scheduler, iteration, current_epoch, 
                   best_metric_value, metric_name, val_loss, config, output_dir):
    """
    save checkpoint and keep only top-k by the tracked metric.
    filename encodes the tracked metric and its value.
    """
    if not accelerator.is_local_main_process:
        return None
    
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    keep_top_k = int(getattr(config.valid.save, "keep_top_k", 1))
    lower_is_better = bool(getattr(config.valid.save, "lower_is_better", True))

    # encode the tracked metric
    filename = f"best_{metric_name}_{best_metric_value:.6f}_iter_{iteration+1}_epoch_{current_epoch+1}.pth"
    checkpoint_path = os.path.join(checkpoints_dir, filename)
    
    # pack state
    checkpoint = {
        "iteration": iteration + 1,
        "epoch": current_epoch + 1,
        "state_dict": accelerator.unwrap_model(model).state_dict(),
        "best_metric": float(best_metric_value),
        "metric_name": metric_name,
        "val_loss": float(val_loss),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Metrics - iteration: {iteration+1} | epoch: {current_epoch+1} | val_loss: {val_loss:.6f} | {metric_name}: {best_metric_value:.6f}")

    # enforce top-k retention
    if keep_top_k > 0:
        entries = []
        for p in _list_checkpoints(checkpoints_dir):
            mname, mval = _parse_metric_from_name(p)
            if mname == metric_name and mval is not None and np.isfinite(mval):
                entries.append((p, mval))
            else:
                # fallback to metadata if needed
                try:
                    meta = torch.load(p, map_location="cpu")
                    if meta.get("metric_name") == metric_name:
                        mv = float(meta.get("best_metric", float("inf")))
                        entries.append((p, mv))
                except Exception:
                    # if unreadable, deprioritize
                    entries.append((p, float("inf") if lower_is_better else float("-inf")))

        # sort and prune
        entries.sort(key=lambda x: x[1], reverse=not lower_is_better)
        for p, _ in entries[keep_top_k:]:
            try:
                os.remove(p)
                print(f"Pruned checkpoint: {p}")
            except OSError as e:
                print(f"Warning: Could not delete {p}: {e}")
    
    return checkpoint_path


import shutil

def create_symlink_for_wids_cache():
    target = "/scratch/km6748/_wids_cache"
    link = "/tmp/_wids_cache"

    os.makedirs(target, exist_ok=True)

    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        shutil.rmtree(link, ignore_errors=True)

    os.symlink(target, link)


class AverageMeter(object):
    """computes and stores the average and current value of metrics."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_last_model(accelerator, model, optimizer, scheduler, iteration, current_epoch, config, output_dir):
    """
    Save the final model checkpoint with the same iter/epoch suffix as best models,
    but without validation metric values.
    """
    if not accelerator.is_local_main_process:
        return None

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    filename = f"last_iter_{iteration+1}_epoch_{current_epoch+1}.pth"
    checkpoint_path = os.path.join(checkpoints_dir, filename)

    checkpoint = {
        "iteration": iteration + 1,
        "epoch": current_epoch + 1,
        "state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": OmegaConf.to_container(config, resolve=True),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved last checkpoint: {checkpoint_path}")
    return checkpoint_path

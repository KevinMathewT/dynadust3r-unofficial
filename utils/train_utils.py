import os
import glob
import wandb
import random
import numpy as np
from accelerate import Accelerator

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

def save_best_model(accelerator, model, optimizer, scheduler, iteration, current_epoch, 
                   best_metric_value, metric_name, val_loss, config, output_dir):
    """
    Save the best model with validation loss in filename and delete prior weights.
    
    Args:
        accelerator: HuggingFace Accelerator instance
        model: The model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save (can be None)
        iteration: Current iteration number
        current_epoch: Current epoch number
        best_metric_value: Best metric value achieved
        metric_name: Name of the metric being tracked
        val_loss: Current validation loss
        config: Configuration object
        output_dir: Directory where to save the checkpoints
    
    Returns:
        str: Path to saved checkpoint file
    """
    if not accelerator.is_local_main_process:
        return None
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Delete all existing checkpoint files
    existing_checkpoints = glob.glob(os.path.join(checkpoints_dir, "best_model_*.pth"))
    for checkpoint_path in existing_checkpoints:
        try:
            os.remove(checkpoint_path)
            print(f"Deleted previous checkpoint: {checkpoint_path}")
        except OSError as e:
            print(f"Warning: Could not delete {checkpoint_path}: {e}")
    
    # Create filename with validation loss
    filename = f"best_model_iter_{iteration+1}_epoch_{current_epoch+1}_val_loss_{val_loss:.6f}.pth"
    checkpoint_path = os.path.join(checkpoints_dir, filename)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        "iteration": iteration + 1,
        "epoch": current_epoch + 1,
        "state_dict": accelerator.unwrap_model(model).state_dict(),
        "best_metric": best_metric_value,
        "metric_name": metric_name,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Saved best model at: {checkpoint_path}")
    print(f"Metrics - iteration: {iteration+1} | epoch: {current_epoch+1} | val_loss: {val_loss:.6f} | {metric_name}: {best_metric_value:.6f}")
    
    return checkpoint_path


import os
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
    """Computes and stores the average and current value of metrics."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation of the meter."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from loaders.pointodyssey import PointOdyssey
from loaders.stereo4d import Stereo4D
from loaders.stereo4d_wds import Stereo4DWDS

LOADERS = {
    "pointodyssey": PointOdyssey,
    # "stereo4d": Stereo4D,
    "stereo4d": Stereo4DWDS,
}

def add_batch_size_wrapper(batch):
    batch = default_collate(batch)
    batch['batch_size'] = len(batch['left_pm'])  # or use any other tensor in the batch
    
    return batch

def get_loaders(config, num_workers_multiplier=1, time_debug=False, loader_kwargs=None):
    """
    Create dataloaders for use with Hugging Face Accelerate.
    
    Important: When using Accelerate, do NOT:
    - Manually create DistributedSampler
    - Divide batch_size by world_size
    - Handle any distributed logic
    
    Accelerate's prepare() method will handle all distributed training setup.
    
    Args:
        config: Configuration object
        num_workers_multiplier: Multiplier for number of workers
        time_debug: Whether to enable timing debug in the dataset
        loader_kwargs: Optional dict forwarded verbatim to DataLoader to allow
                        quick sweeps of parameters like prefetch_factor &
                        persistent_workers without touching this function.
    """
    # Create datasets with time_debug parameter
    train_dataset = LOADERS[config.data.loader](config, valid=False, time_debug=time_debug)
    valid_dataset = LOADERS[config.data.loader](config, valid=True, time_debug=time_debug)

    # Calculate num_workers based on multiplier
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_workers = max(1, int(num_workers_multiplier * len(os.sched_getaffinity(0)) // world_size))

    # ensure dict so that .get works even if None passed
    loader_kwargs = loader_kwargs or {}

    # derive final values (dict overrides the auto‑defaults when supplied)
    persistent_workers = True # loader_kwargs.get('persistent_workers', True if num_workers > 0 else False)
    prefetch_factor    = loader_kwargs.get('prefetch_factor', 8 if num_workers > 0 else None)

    # Create dataloaders with FULL batch size
    # Accelerate will automatically handle splitting across GPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,  # Use full batch size
        shuffle=True,  # Always shuffle for training
        num_workers=num_workers,
        collate_fn=add_batch_size_wrapper,
        pin_memory=True,  # Recommended for GPU training
        persistent_workers=persistent_workers,  # Keeps workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch batches for better performance
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.batch_size * 4,  # Use full batch size
        shuffle=False,  # Never shuffle validation
        num_workers=num_workers,
        collate_fn=add_batch_size_wrapper,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor * 2,  # Prefetch batches for better performance
    )

    # Only print info if we're on the main process
    # Note: This check is safe because it happens after setup_distributed() in train.py
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    if is_main:
        print(f"world_size: {world_size}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        print(f"Train dataloader batches: {len(train_loader)}")
        print(f"Validation dataloader batches: {len(valid_loader)}")
        print(f"Batch size (before distribution): {config.data.batch_size}")
        print(f"num_workers: {num_workers} ({num_workers_multiplier} * {len(os.sched_getaffinity(0))})")
        print(f"persistent_workers: {persistent_workers} | prefetch_factor: {prefetch_factor}")
        print(f"time_debug: {time_debug}")
        print(f"Note: Accelerate will automatically distribute batches across GPUs")

    return train_loader, valid_loader


# -------------------------------------------------------------
#  dataloader benchmark – mirrors dynadust3r config handling
# -------------------------------------------------------------
import os, time, itertools
import torch, hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm                                                   # NEW

from utils.train_utils import setup_distributed, AverageMeter
from loaders import get_loaders           # keep import path same as your project


def get_cycled_batches(dataloader, total_iterations):
    """yield exactly total_iterations batches, cycling through the dataloader"""
    it = 0
    while it < total_iterations:
        for batch in dataloader:
            yield batch
            it += 1
            if it >= total_iterations:
                return


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    """benchmarks dataloader throughput for a fixed 200 iterations, with tqdm"""
    # debug override – comment out if not wanted
    # config.data.batch_size= 8
    # config.data.len       = 1600
    # config.data.valid_len = 1600

    setup_distributed(config.seed)
    is_main = (not torch.distributed.is_initialized() or
               torch.distributed.get_rank() == 0)

    if is_main:
        print("----- config -----")
        print(OmegaConf.to_yaml(config))
        print("------------------")

    output_dir = HydraConfig.get().runtime.output_dir
    if is_main:
        print("output directory:", output_dir)

    # Experiment with different num_workers multipliers
    multipliers_to_test = [0, 0.25, 0.5, 0.75, 1, 2, 3, 4]
    # NEW: sweep prefetch_factor & persistent_workers
    prefetch_values    = [None, 2, 4, 8, 16]
    persistent_options = [False, True]

    results = {}

    # Enable time_debug for detailed timing analysis
    time_debug = True
    
    for multiplier in multipliers_to_test:
        for pref in prefetch_values:
            for persist in persistent_options:

                num_workers = int(multiplier * os.cpu_count() / 3)
                if num_workers == 0 and persist:
                    continue  # illegal combination

                tag = f"x{multiplier}_pf{pref}_pw{int(persist)}"
                if is_main:
                    print(f"\n{'='*50}\nTesting {tag}\n{'='*50}")

                loader_kwargs = dict(prefetch_factor=pref, persistent_workers=persist)
                train_loader, _ = get_loaders(config, num_workers_multiplier=multiplier,
                                              time_debug=time_debug, loader_kwargs=loader_kwargs)

                # fixed 200 iterations, regardless of epoch boundaries
                total_iterations = 20
                if is_main:
                    print(f"benchmarking exactly {total_iterations} iterations")

                load_time = AverageMeter("load_time", ":.6f")

                torch.manual_seed(config.seed)

                # tqdm progress bar (main process only)
                pbar = tqdm(total=total_iterations, disable=not is_main, ncols=80, 
                           desc=f"benchmark ({tag})")

                batch_iterator = get_cycled_batches(train_loader, total_iterations)
                for iteration in range(total_iterations):
                    start = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    batch = next(batch_iterator)                            # time the actual batch loading

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end = time.perf_counter()

                    _ = batch["left_image"][0]                              # touch data to ensure it's loaded

                    load_time.update(end - start, 1)
                    pbar.set_postfix(cur=f"{load_time.val:.4f}s", avg=f"{load_time.avg:.4f}s")
                    pbar.update(1)

                pbar.close()

                # Store results
                results[(multiplier, pref, persist)] = {
                    'avg': load_time.avg
                }

                if is_main:
                    print(f"\n=== Results for {tag} ===")
                    print(f"iterations:        {total_iterations}")
                    print(f"avg load / batch:  {load_time.avg:.6f}s")

    # Print summary of all results
    if is_main:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY - num_workers/prefetch/persistent sweep")
        print(f"{'='*60}")
        print(f"{'mult':<8} {'workers':<8} {'pref':<6} {'persist':<8} {'avg':<10}")
        print("-" * 44)
        
        for (multiplier, pref, persist), v in results.items():
            num_workers = int(multiplier * os.cpu_count() / 3)
            print(f"{multiplier:<8} {num_workers:<8} {str(pref):<6} {str(persist):<8} {v['avg']:<10.6f}")
        
        # Find best performing configuration
        best_key = min(results.keys(), key=lambda k: results[k]['avg'])
        b_mult, b_pref, b_persist = best_key
        best_workers = int(b_mult * os.cpu_count() / 3)
        best_time = results[best_key]['avg']
        
        print(f"\nBEST CONFIGURATION:")
        print(f"multiplier        : {b_mult}")
        print(f"num_workers       : {best_workers}")
        print(f"prefetch_factor   : {b_pref}")
        print(f"persistent_workers: {b_persist}")
        print(f"average / batch   : {best_time:.6f}s")
        print(f"{'='*60}")


from utils.train_utils import (
    setup_distributed, init_wandb, save_best_model,
    create_symlink_for_wids_cache, AverageMeter,
)
if __name__ == "__main__":
    create_symlink_for_wids_cache()
    main()

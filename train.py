#!/usr/bin/env python
# coding: utf-8
"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

# -------------------------------------------------------------
#  DynaDUSt3R training script
# -------------------------------------------------------------
import os
import time
import torch, torch.nn.functional as F
import wandb, numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta

from models import get_model
from loaders import get_loaders
from optimizer import get_optimizer, get_scheduler
from utils.train_utils import (
    setup_distributed, init_wandb, save_best_model,
    create_symlink_for_wids_cache, AverageMeter,
)
import utils.wandb_utils as wandb_logger


def get_cycled_batches(dataloader, accelerator, total_iterations):
    """yields batches, cycling through dataloader as needed"""
    iteration = 0
    while iteration < total_iterations:
        for batch in dataloader:
            yield batch
            iteration += 1
            if iteration >= total_iterations:
                return
        accelerator.wait_for_everyone()  # keep epochs aligned across ranks


# ═════════════════════════════════════════════════════════════
#                     H Y D R A   M A I N
# ═════════════════════════════════════════════════════════════
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):

    if config.debug:
        # overfitting experiment on tiny debug dataset
        config.data.len                   = 4
        config.data.valid_len             = 4
        config.data.batch_size            = 4
        config.train.validation_frequency = 10
        train_viz_interval                = 5
    else:
        # align dataset sizes with training plan
        config.data.len                   = config.train.iterations * config.data.batch_size
        config.data.valid_len             = config.data.valid_len * config.data.batch_size
        train_viz_interval                = 250

    setup_distributed(config.seed)
    is_main = (not torch.distributed.is_initialized() or
               torch.distributed.get_rank() == 0)

    output_dir = HydraConfig.get().runtime.output_dir

    # allow params skipped in some forwards
    ddp_kwargs   = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs  = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator  = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

    # wandb
    if config.logging.use_wandb and accelerator.is_local_main_process:
        init_wandb(OmegaConf.to_container(config, resolve=True))

    if accelerator.is_local_main_process:
        create_symlink_for_wids_cache()

    model                      = get_model(config, accelerator.device)
    train_loader, valid_loader = get_loaders(config)
    optimizer                  = get_optimizer(model.parameters(), config)

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    scheduler = get_scheduler(optimizer, config, train_loader)

    # training setup
    model.train()
    train_losses  = AverageMeter("Loss", ":.4e")
    train_metrics = {}

    total_iterations      = config.train.iterations
    validation_frequency  = config.train.validation_frequency
    batches_per_epoch     = len(train_loader)

    batch_generator = get_cycled_batches(train_loader, accelerator, total_iterations)
    batch_iter      = batch_generator
    iteration       = 0

    start_time = time.time()  # wall-clock timer

    # main training loop
    for batch in batch_iter:
        current_epoch     = iteration // validation_frequency
        dataset_position  = (iteration % batches_per_epoch) + 1

        outputs = model(batch)  # (depends on model)  # (..., ...)

        # loss in fp32 to avoid nans in some terms
        unwrapped_model = accelerator.unwrap_model(model)
        loss, loss_details = unwrapped_model.get_loss(batch, outputs)  # scalar tensor  # (1,)

        if loss is None or not isinstance(loss, torch.Tensor):
            loss = torch.zeros(1, requires_grad=True, device=accelerator.device)  # (1,)

        # backward
        accelerator.backward(loss)

        # grad clip
        if config.train.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        else:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

        optimizer.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients and scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # metrics on-no-grad
        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)
            batch_metrics = unwrapped_model.compute_metrics(batch, outputs)  # dict[str, float]

            for k, v in batch_metrics.items():
                if k not in train_metrics:
                    train_metrics[k] = AverageMeter(k, ":.4f")
                train_metrics[k].update(v, batch["batch_size"])

        train_losses.update(loss.item(), batch["batch_size"])

        if (iteration % 5 == 0 or iteration == total_iterations - 1):
            elapsed   = time.time() - start_time
            done_frac = (iteration + 1) / total_iterations
            remaining = elapsed * (1 - done_frac) / done_frac if done_frac > 0 else 0
            days  = int(remaining // 86400); remaining %= 86400
            hrs   = int(remaining // 3600);  remaining %= 3600
            mins  = int(remaining // 60);    secs = int(remaining % 60)
            eta_str = f"{days}d {hrs}h {mins}m {secs}s"

            lr = optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
            local_bs  = len(batch["left_image"])
            global_bs = local_bs * accelerator.num_processes
            accelerator.print(
                f"[{current_epoch+1}][{iteration+1}/{total_iterations}]"
                f"[{dataset_position}/{batches_per_epoch}] "
                f"train loss: {loss.item():.10f} | lr: {lr:.10f} | "
                f"grad norm: {grad_norm} | batch size: {local_bs} "
                f"(global: {global_bs}) | eta: {eta_str}"
            )

        if iteration % config.logging.wandb_interval == 0 and accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]["lr"]
            wandb_logger.log_training_batch(
                iteration, current_epoch, loss.item(), lr, loss_details, train_metrics, config, accelerator
            )

        if iteration % train_viz_interval == 0 and accelerator.is_local_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            base_name = f"t_e_{current_epoch}_b_{iteration}"
            unwrapped_model.save_visualizations(batch, outputs, base_name)  # imgs/grids  # (B,H,W,3)

        if iteration % train_viz_interval == 0:
            accelerator.wait_for_everyone()  # barrier after rank-0 training viz

        # validation
        if (iteration + 1) % validation_frequency == 0 or iteration == total_iterations - 1:
            if accelerator.is_local_main_process:
                wandb_logger.log_training_epoch(
                    iteration, current_epoch, train_losses, train_metrics, optimizer, config, accelerator
                )

            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] train epoch loss: {train_losses.avg:.10f}")

            train_metrics_str = ""
            for k, meter in train_metrics.items():
                train_metrics_str += f" | {k}: {meter.avg:.3f}"
            if train_metrics_str:
                accelerator.print(f"[{current_epoch+1}] train{train_metrics_str}")

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_name = config.sched.metric
                if metric_name == "loss":
                    scheduler.step(train_losses.avg)
                elif metric_name in train_metrics:
                    scheduler.step(train_metrics[metric_name].avg)

            model.eval()
            val_losses  = AverageMeter("Loss", ":.4e")
            val_metrics = {}

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(valid_loader):
                    val_outputs = model(val_batch)  # (..., ...)

                    unwrapped_model = accelerator.unwrap_model(model)
                    val_loss, val_loss_details = unwrapped_model.get_loss(val_batch, val_outputs)  # (1,)

                    if val_loss is None or not isinstance(val_loss, torch.Tensor):
                        val_loss = torch.zeros(1, device=accelerator.device)  # (1,)

                    val_batch_metrics = unwrapped_model.compute_metrics(val_batch, val_outputs)

                    for k, v in val_batch_metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = AverageMeter(k, ":.4f")
                        val_metrics[k].update(v, val_batch["batch_size"])

                    val_losses.update(val_loss.item(), val_batch["batch_size"])

                    if val_batch_idx % config.logging.wandb_interval == 0 and accelerator.is_local_main_process:
                        wandb_logger.log_validation_batch(
                            iteration, current_epoch, val_batch_idx, val_loss.item(),
                            val_loss_details, val_batch_metrics, config, accelerator
                        )

                    if (val_batch_idx == 0 or (val_batch_idx + 1) % 5 == 0 or val_batch_idx == len(valid_loader) - 1):
                        accelerator.print(
                            f"[{current_epoch+1}][{iteration+1}/{total_iterations}][{val_batch_idx+1}/{len(valid_loader)}] valid loss: {val_loss.item():.10f}"
                        )

                if accelerator.is_local_main_process:
                    valid_vis_dir = os.path.join(output_dir, "valid")
                    os.makedirs(valid_vis_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    base_name = f"v_e_{current_epoch}_b_{iteration}"
                    unwrapped_model.save_visualizations(val_batch, val_outputs, base_name)  # (B,H,W,3)
                accelerator.wait_for_everyone()  # barrier after rank-0 validation viz

            if accelerator.is_local_main_process:
                wandb_logger.log_validation_epoch(iteration, current_epoch, val_losses, val_metrics, config, accelerator)

            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] val epoch loss: {val_losses.avg:.10f}")

            val_metrics_str = ""
            for k, meter in val_metrics.items():
                val_metrics_str += f" | {k}: {meter.avg:.3f}"
            if val_metrics_str:
                accelerator.print(f"[{current_epoch+1}] val{val_metrics_str}")

            # best-checkpoint save; keep_top_k>1 => save every val, prune in saver
            if config.valid.save.save_best:
                metric_name = config.valid.save.best_metric
                keep_top_k = int(getattr(config.valid.save, "keep_top_k", 1))

                if metric_name == "loss":
                    metric_val = val_losses.avg
                    better = lambda new, old: new < old
                elif metric_name in val_metrics:
                    metric_val = val_metrics[metric_name].avg
                    better = lambda new, old: (new < old if config.valid.save.lower_is_better else new > old)
                else:
                    accelerator.print(f"warning: best_metric '{metric_name}' not found in metrics.")
                    metric_val = None

                if metric_val is not None and accelerator.is_local_main_process:
                    should_save = False
                    if keep_top_k > 1:
                        should_save = True  # saver will prune to top-k
                    else:
                        unwrapped_model = accelerator.unwrap_model(model)
                        if not hasattr(unwrapped_model, "best_metric"):
                            unwrapped_model.best_metric = float("inf") if better(0, 1) else float("-inf")
                            should_save = True
                        elif better(metric_val, unwrapped_model.best_metric):
                            unwrapped_model.best_metric = metric_val
                            should_save = True

                    if should_save:
                        save_best_model(
                            accelerator, model, optimizer, scheduler, iteration, current_epoch,
                            float(metric_val), metric_name, val_losses.avg, config, output_dir
                        )

                accelerator.wait_for_everyone()  # keep ranks in sync leaving validation

            train_losses.reset()
            for k, meter in train_metrics.items():
                meter.reset()

            model.train()

        iteration += 1

    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()

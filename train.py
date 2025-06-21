import sys
import pdb

def custom_excepthook(type, value, traceback):
    print("\n\n--- entering post-mortem debugging ---\n")
    pdb.post_mortem(traceback)

sys.excepthook = custom_excepthook

import yaml
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import time
import os
import datetime
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import hydra
from hydra import initialize
from pprint import pprint
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from models import get_model
from loaders import get_loaders
from criterion import get_criterion
from optimizer import get_optimizer, get_scheduler
from utils.train_utils import setup_distributed, init_wandb, save_best_model, create_symlink_for_wids_cache, AverageMeter
import utils.wandb_utils as wandb_logger


# Load config
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    config.data.len = config.train.iterations * config.data.batch_size
    config.data.valid_len = config.data.valid_len * config.data.batch_size

    print(f"----- Config -----")
    print(OmegaConf.to_yaml(config))
    print(f"-----------------")

    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # Initialize wandb
    if config.logging.use_wandb:
        init_wandb(OmegaConf.to_container(config, resolve=True))

    # Setup
    setup_distributed(config.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if accelerator.is_local_main_process:
        create_symlink_for_wids_cache()

    model = get_model(config, accelerator.device)
    train_loader, valid_loader = get_loaders(config)

    # Initialize model, criterion, optimizer, scheduler
    criterion = get_criterion(config)
    optimizer = get_optimizer(model.parameters(), config)

    accelerator.print(f"----- Model -----")
    accelerator.print(model)
    accelerator.print(f"-----------------")

    accelerator.print(f"--------- Criterion, Optimizer ---------")
    accelerator.print(f"criterion: {criterion}")
    accelerator.print(f"optimizer: {optimizer}")
    accelerator.print(f"----------------------------------------")

    # Prepare everything with accelerator
    model, optimizer, criterion, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, criterion, train_loader, valid_loader
    )

    scheduler = get_scheduler(optimizer, config, train_loader)
    accelerator.print(f"---------------- Scheduler ----------------")
    accelerator.print(f"scheduler: {scheduler}")
    accelerator.print(f"-------------------------------------------")

    # Training setup
    model.train()
    train_losses = AverageMeter("Loss", ":.4e")
    train_metrics = {}
    
    # Use separate parameters for total iterations vs dataset length
    total_iterations = config.train.iterations  # Total training iterations
    dataset_length = config.data.len  # Dataset length for cycling
    validation_frequency = config.train.validation_frequency  # How often to validate (replaces both epoch_frequency and valid.interval)

    accelerator.print(f"Training for {total_iterations} iterations with dataset cycling every {dataset_length} iterations")
    accelerator.print(f"Validation every {validation_frequency} iterations")

    # Create infinite iterator from train_loader
    train_iter = iter(train_loader)
    dataset_iteration_count = 0  # Track iterations within current dataset cycle

    # Main training loop - iterate for total_iterations
    for iteration in range(total_iterations):
        # Check if we need to reset the dataset (cycle after data.len iterations)
        if dataset_iteration_count >= dataset_length:
            accelerator.print(f"Cycling dataset at iteration {iteration} (after {dataset_iteration_count} dataset iterations)")
            train_iter = iter(train_loader)
            dataset_iteration_count = 0
        
        # Get next batch
        try:
            batch = next(train_iter)
            dataset_iteration_count += 1
        except StopIteration:
            # Fallback in case StopIteration occurs before dataset_length
            accelerator.print(f"StopIteration occurred at iteration {iteration}, resetting iterator")
            train_iter = iter(train_loader)
            batch = next(train_iter)
            dataset_iteration_count = 1
        
        # Calculate current "epoch" for logging purposes
        current_epoch = iteration // validation_frequency
        
        # forward prop
        with accelerator.autocast():
            outputs = model(batch)
            loss, loss_details = model.module.get_loss(criterion, batch, outputs)

        # backward prop
        if isinstance(loss, torch.Tensor):
            accelerator.backward(loss)
        else:
            print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] GOT ZERO LOSS; which means the valid mask is not valid anywhere, check mask sum below:")
            for i in range(batch["left_image"].size(0)):
                print(f'get_loss | {i} | instance1: {batch["left_instance"][i]} | instance2: {batch["right_instance"][i]}')
                print(f'get_loss | {i} | mask1 sum: {batch["left_pm"][i][..., 3].sum()} | mask2 sum: {batch["right_pm"][i][..., 3].sum()} | og shape: {batch["left_pm"][i].shape}')

            loss = torch.zeros(1).to(batch["left_image"].device)

        # grad clip
        if config.train.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)

        # update weights
        optimizer.step()
        if accelerator.sync_gradients and scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # compute metrics
        with torch.no_grad():
            batch_metrics = model.module.compute_metrics(batch, outputs)

            for k, v in batch_metrics.items():
                if k not in train_metrics:
                    train_metrics[k] = AverageMeter(k, ":.4f")
                train_metrics[k].update(v, batch["batch_size"])

        train_losses.update(loss.item(), batch["batch_size"])

        # batch-level logging
        if (iteration % 5 == 0 or iteration == total_iterations - 1):
            lr = optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
            accelerator.print(
                f"[{current_epoch+1}][{iteration+1}/{total_iterations}][{dataset_iteration_count}/{dataset_length}] train loss: {loss.item():.10f} | lr: {lr:.10f} | grad norm: {grad_norm} |",
                end="",
            )
            # for k, v in loss_details.items():
            #     accelerator.print(f" {k}: {v:.4f}", end=" |")
            accelerator.print("")

        # wandb logging
        if iteration % config.logging.wandb_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            wandb_logger.log_training_batch(iteration, current_epoch, loss.item(), lr, loss_details, train_metrics, config, accelerator)

        # Validation and epoch-level summaries (every validation_frequency iterations)
        if (iteration + 1) % validation_frequency == 0 or iteration == total_iterations - 1:
            # epoch-level logging to wandb
            wandb_logger.log_training_epoch(iteration, current_epoch, train_losses, train_metrics, optimizer, config, accelerator)

            # epoch summary for training
            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] train epoch loss: {train_losses.avg:.10f}")

            train_metrics_str = ""
            for k, meter in train_metrics.items():
                train_metrics_str += f" | {k}: {meter.avg:.3f}"

            if train_metrics_str:
                accelerator.print(f"[{current_epoch+1}] train{train_metrics_str}")

            # Save training visualizations (every 5 validation cycles to avoid too many)
            if current_epoch % 5 == 0:
                model.module.save_visualizations(batch, outputs, current_epoch, iteration, output_dir, criterion)

            # lr scheduler step (for ReduceLROnPlateau)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_name = config.sched.metric
                if metric_name == "loss":
                    scheduler.step(train_losses.avg)
                elif metric_name in train_metrics:
                    scheduler.step(train_metrics[metric_name].avg)

            # VALIDATION
            model.eval()
            val_losses = AverageMeter("Loss", ":.4e")
            val_metrics = {}

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(valid_loader):
                    # forward prop
                    with accelerator.autocast():
                        val_outputs = model(val_batch)
                        val_loss, val_loss_details = model.module.get_loss(criterion, val_batch, val_outputs)

                    # compute metrics
                    val_batch_metrics = model.module.compute_metrics(val_batch, val_outputs)

                    for k, v in val_batch_metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = AverageMeter(k, ":.4f")
                        val_metrics[k].update(v, val_batch["batch_size"])

                    val_losses.update(val_loss.item(), val_batch["batch_size"])

                    # Add validation batch logging here
                    if val_batch_idx % config.logging.wandb_interval == 0:
                        wandb_logger.log_validation_batch(
                            iteration, current_epoch, val_batch_idx, val_loss.item(), 
                            val_loss_details, val_batch_metrics, config, accelerator
                        )

                    # logging
                    if (val_batch_idx == 0 or (val_batch_idx + 1) % 5 == 0 or val_batch_idx == len(valid_loader) - 1):
                        accelerator.print(
                            f"[{current_epoch+1}][{iteration+1}/{total_iterations}][{val_batch_idx+1}/{len(valid_loader)}] valid loss: {val_loss.item():.10f}"
                        )

                valid_vis_dir = os.path.join(output_dir, "valid")
                os.makedirs(valid_vis_dir, exist_ok=True)
                model.module.save_visualizations(val_batch, val_outputs, current_epoch, iteration, valid_vis_dir, criterion)

            # wandb logging for validation
            wandb_logger.log_validation_epoch(iteration, current_epoch, val_losses, val_metrics, config, accelerator)

            # epoch summary for validation
            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] val epoch loss: {val_losses.avg:.10f}")

            val_metrics_str = ""
            for k, meter in val_metrics.items():
                val_metrics_str += f" | {k}: {meter.avg:.3f}"

            if val_metrics_str:
                accelerator.print(f"[{current_epoch+1}] val{val_metrics_str}")

            # save best model
            if config.valid.save.save_best:
                is_best = False
                metric_name = config.valid.save.best_metric

                if metric_name == "loss":
                    metric_val = val_losses.avg
                    better = lambda new, old: new < old
                elif metric_name in val_metrics:
                    metric_val = val_metrics[metric_name].avg
                    better = lambda new, old: (new < old if config.valid.save.lower_is_better else new > old)
                else:
                    accelerator.print(f"warning: best_metric '{metric_name}' not found in metrics.")
                    continue

                if not hasattr(model.module, "best_metric"):
                    model.module.best_metric = float("inf") if better(0, 1) else float("-inf")
                    is_best = True
                elif better(metric_val, model.module.best_metric):
                    is_best = True
                    model.module.best_metric = metric_val

                if is_best:
                    checkpoint_path = save_best_model(
                        accelerator, model, optimizer, scheduler, iteration, current_epoch,
                        model.module.best_metric, metric_name, val_losses.avg, config, output_dir
                    )

            # Reset training metrics for next epoch period
            train_losses.reset()
            for k, meter in train_metrics.items():
                meter.reset()

            model.train()

    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()
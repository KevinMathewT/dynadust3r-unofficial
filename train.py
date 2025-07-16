#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------
#  DynaDUSt3R training script  –  now with MemLab + Profiler
# -------------------------------------------------------------
import os
import sys
import pdb

# ──────────────────────────────────────────────────────────────
# post-mortem only on rank-0
# ──────────────────────────────────────────────────────────────
def custom_excepthook(type, value, traceback):
    print("\n\n--- entering post-mortem debugging ---\n")
    pdb.post_mortem(traceback)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:   # only on the main GPU
    sys.excepthook = custom_excepthook


# ──────────────────────────────────────────────────────────────
# stdlib / 3rd-party imports
# ──────────────────────────────────────────────────────────────
import yaml, itertools, time, datetime
import torch, torch.nn.functional as F
import wandb, numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from models import get_model
from loaders import get_loaders
from criterion import get_criterion
from optimizer import get_optimizer, get_scheduler
from utils.train_utils import (
    setup_distributed, init_wandb, save_best_model,
    create_symlink_for_wids_cache, AverageMeter,
)
import utils.wandb_utils as wandb_logger


# ──────────────────────────────────────────────────────────────
# ### ──────────────── mem / prof additions ──────────────── ###
# ──────────────────────────────────────────────────────────────
# memlab - import is optional
try:
    from pytorch_memlab import MemReporter          # v0.3+  (MemCounter was removed)
    HAS_MEMLAB = True
except Exception as e:
    HAS_MEMLAB = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("[memlab] not available – continuing without MemReporter ::", e)

# profiler – enable with  USE_PROFILER=1
USE_PROFILER = os.environ.get("USE_PROFILER", "0") == "1"
if USE_PROFILER:
    from torch.profiler import (
        profile, schedule, ProfilerActivity, tensorboard_trace_handler
    )
# ──────────────────────────────────────────────────────────────


def get_cycled_batches(dataloader, accelerator, total_iterations):
    """Yields batches, cycling through dataloader as needed"""
    iteration = 0
    while iteration < total_iterations:
        for batch in dataloader:
            yield batch
            iteration += 1
            if iteration >= total_iterations:
                return
        # Synchronize before starting new cycle
        accelerator.wait_for_everyone()
        # accelerator.print(f"Cycling dataset at iteration {iteration}")

def install_nan_hook(net):
    def has_non_finite(x):
        if torch.is_tensor(x):
            return (~torch.isfinite(x)).any()
        elif isinstance(x, (list, tuple)):
            return any(has_non_finite(t) for t in x)
        elif isinstance(x, dict):
            return any(has_non_finite(t) for t in x.values())
        else:
            return False           # ignore scalars/None/other types

    def _hook(mod, inp, out):
        if has_non_finite(inp) or has_non_finite(out):
            raise RuntimeError(f"🛑 NaN/Inf detected in {mod.__class__.__name__}")

    for m in net.modules():
        m.register_forward_hook(_hook)


def install_nan_hook_verbose(net):
    def stats(t):
        return dict(
            min=float(t.min()), max=float(t.max()),
            mean=float(t.mean()), std=float(t.std()),
            n_nan=int((~torch.isfinite(t)).sum()),
        )

    def _hook(mod, inp, out):
        # Flatten possible tuple / list structures
        xs = [*inp]
        if not isinstance(out, (tuple, list)):
            xs.append(out)
        else:
            xs.extend(out)

        # Collect stats
        finites = [x for x in xs if torch.is_tensor(x)]
        any_bad = any((~torch.isfinite(x)).any() for x in finites)
        if any_bad:
            print(f"\n🛑  NaN/Inf inside: {mod.__class__.__name__}")
            for i, x in enumerate(finites):
                print(f"   tensor {i}: {stats(x)}")
            raise RuntimeError("stopping on first non-finite value")

    for m in net.modules():
        m.register_forward_hook(_hook)


# ═════════════════════════════════════════════════════════════
#                     H Y D R A   M A I N
# ═════════════════════════════════════════════════════════════
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    global HAS_MEMLAB, USE_PROFILER
    HAS_MEMLAB = HAS_MEMLAB and config.debug  # enable memlab only in debug mode
    USE_PROFILER = USE_PROFILER and config.debug # enable profiler only in debug mode

    # tiny debug dataset
    config.data.len        = 1
    config.data.valid_len  = 1

    # actual dataset
    # config.data.len        = config.train.iterations * config.data.batch_size
    # config.data.valid_len  = config.data.valid_len * config.data.batch_size

    # seed & distributed
    setup_distributed(config.seed)
    is_main = (not torch.distributed.is_initialized() or
               torch.distributed.get_rank() == 0)

    if is_main:
        print(f"HAS_MEMLAB: {HAS_MEMLAB}")
        print(f"USE_PROFILER: {USE_PROFILER}")
        print("----- Config -----")
        print(OmegaConf.to_yaml(config))
        print("------------------")

    output_dir = HydraConfig.get().runtime.output_dir
    if is_main:
        print("Output directory:", output_dir)

    # accelerator
    # ddp_kwargs   = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator  = Accelerator() # kwargs_handlers=[ddp_kwargs])
    print(">>> Accelerate mixed_precision =", accelerator.mixed_precision)

    pid = os.getpid()
    rank = accelerator.process_index  # 0, 1, 2, … for each process
    accelerator.print(f"rank {rank} pid: {pid}") # this will print one line per GPU process

    # wandb
    if config.logging.use_wandb and accelerator.is_local_main_process:
        init_wandb(OmegaConf.to_container(config, resolve=True))

    if accelerator.is_local_main_process:
        create_symlink_for_wids_cache()

    # data / model
    model                      = get_model(config, accelerator.device)
    train_loader, valid_loader = get_loaders(config)
    criterion                  = get_criterion(config)
    optimizer                  = get_optimizer(model.parameters(), config)

    # install_nan_hook(model)        # <<<<<<<<<<<<<<

    # Check for non-finite weights in model parameters
    for n, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"non-finite weight {n}")


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

    install_nan_hook_verbose(model)
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"NaN/Inf in weight {n} AFTER prepare")

    
    # one-shot report of parameters & buffers
    if HAS_MEMLAB and accelerator.is_local_main_process:
        MemReporter(accelerator.unwrap_model(model)).report()

    # profiler (rank-0 only)
    profiler = None
    if USE_PROFILER and accelerator.is_local_main_process:
        profiler = profile(
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler(os.path.join(output_dir, "tb_prof")),
            with_stack=True,
        )
        profiler.start()

    scheduler = get_scheduler(optimizer, config, train_loader)
    accelerator.print(f"---------------- Scheduler ----------------")
    accelerator.print(f"scheduler: {scheduler}")
    accelerator.print(f"-------------------------------------------")

    # Log distributed training information
    accelerator.print(f"\n========== Distributed Training Info ==========")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Process index: {accelerator.process_index}")
    accelerator.print(f"Local process index: {accelerator.local_process_index}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Check actual batch sizes after prepare()
    accelerator.print(f"\n========== Batch Size Information ==========")
    accelerator.print(f"Config batch size (global): {config.data.batch_size}")
    
    # Get a sample batch to check local batch size
    sample_train_iter = iter(train_loader)
    sample_batch = next(sample_train_iter)
    local_batch_size = len(sample_batch['left_image'])
    
    # Calculate global batch size
    global_batch_size = local_batch_size * accelerator.num_processes
    
    accelerator.print(f"Local batch size per GPU: {local_batch_size}")
    accelerator.print(f"Calculated global batch size: {global_batch_size}")
    accelerator.print(f"Train dataloader length: {len(train_loader)}")
    accelerator.print(f"Total train samples per epoch: {len(train_loader) * local_batch_size}")
    
    # Validation info
    val_iter = iter(valid_loader)
    val_sample_batch = next(val_iter)
    val_local_batch_size = len(val_sample_batch['left_image'])
    
    accelerator.print(f"\nValidation:")
    accelerator.print(f"Local batch size per GPU: {val_local_batch_size}")
    accelerator.print(f"Valid dataloader length: {len(valid_loader)}")
    accelerator.print(f"Total valid samples: {len(valid_loader) * val_local_batch_size}")
    accelerator.print(f"==============================================\n")

    # Training setup
    model.train()
    train_losses = AverageMeter("Loss", ":.4e")
    train_metrics = {}

    # Use separate parameters for total iterations vs dataset length
    total_iterations = config.train.iterations  # Total training iterations
    validation_frequency = config.train.validation_frequency  # How often to validate
    batches_per_epoch = len(train_loader)  # For logging purposes

    accelerator.print(f"Training for {total_iterations} iterations")
    accelerator.print(f"Dataset contains {batches_per_epoch} batches")
    accelerator.print(f"Validation every {validation_frequency} iterations")

    # Create batch generator
    batch_generator = get_cycled_batches(train_loader, accelerator, total_iterations)

    # Debug: Show data distribution across GPUs (first batch only)
    if config.debug and accelerator.num_processes > 1:
        first_batch = next(batch_generator)
        
        # Gather instances from all processes
        all_instances = accelerator.gather_for_metrics(first_batch["left_instance"])
        
        accelerator.print(f"\n========== First Batch Distribution ==========")
        accelerator.print(f"Process {accelerator.process_index} has instances: {first_batch['left_instance']}")
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            accelerator.print(f"\nAll instances in first batch across all GPUs:")
            for i, instance in enumerate(all_instances):
                accelerator.print(f"  GPU {i % accelerator.num_processes}: {instance}")
        accelerator.print(f"==============================================\n")
        
        # Process the first batch
        iteration = 0
        batch = first_batch
    else:
        first_batch = None
        iteration = 0
        batch = None

    # ------------------------------------------------------------------
    # iterate lazily: first the already-fetched batch, then the generator
    # ------------------------------------------------------------------
    batch_iter = itertools.chain([first_batch], batch_generator) if first_batch is not None else batch_generator

    start_time = time.time()                         # wall-clock timer
    
    # Main training loop
    for batch in batch_iter:
        # Calculate current "epoch" for logging purposes
        current_epoch = iteration // validation_frequency
        dataset_position = (iteration % batches_per_epoch) + 1

        # if not torch.isfinite(batch["left_image"]).all():
        #     bad = (~torch.isfinite(batch["left_image"])).flatten().nonzero()[:5]
        #     raise RuntimeError(f"Non-finite pixels in left_image at {bad}")
        # if not torch.isfinite(batch["right_image"]).all():
        #     bad = (~torch.isfinite(batch["right_image"])).flatten().nonzero()[:5]
        #     raise RuntimeError(f"Non-finite pixels in right_image at {bad}")

        # for side in ("left_image", "right_image"):
        #     img = batch[side]
        #     assert img.min() >= -1.1 and img.max() <= 1.1, f"{side} range {img.min()}..{img.max()}"

        # # just before model(batch)
        # w = model.patch_embed.proj.weight.data
        # print("patch_embed weight stats:",
        #     w.min().item(), w.max().item(), w.mean().item(), w.std().item())

        # x = batch["left_image"]
        # print("left_image stats:", x.min().item(), x.max().item())

        # with torch.autograd.detect_anomaly():
        outputs = model(batch)
        
        # Loss computation in FP32 to prevent NaN with ConfLoss
        unwrapped_model = accelerator.unwrap_model(model)
        loss, loss_details = unwrapped_model.get_loss(criterion, batch, outputs)

        # Handle loss properly
        if loss is None or not isinstance(loss, torch.Tensor):
            print(f"[Warning] Got None or non-tensor loss at iteration {iteration} | Loss: {loss}")
            # Create a proper zero loss that requires grad
            loss = torch.zeros(1, requires_grad=True, device=accelerator.device)
            
            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] GOT ZERO/NULL LOSS")
            # Log details only on main process to avoid clutter
            if accelerator.is_local_main_process:
                for i in range(batch["left_image"].size(0)):
                    accelerator.print(f'get_loss | {i} | instance1: {batch["left_instance"][i]} | instance2: {batch["right_instance"][i]}')
                    accelerator.print(f'get_loss | {i} | mask1 sum: {batch["left_pm"][i][..., 3].sum()} | mask2 sum: {batch["right_pm"][i][..., 3].sum()} | og shape: {batch["left_pm"][i].shape}')

        # Backward prop - accelerator handles gradient scaling for mixed precision
        accelerator.backward(loss)

        # Gradient clipping - use accelerator's method if available, with proper fallback
        if config.train.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        else:
            # Still use accelerator's method for consistency
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

        # Update weights
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients AFTER optimizer step (more efficient)
        
        # Step scheduler - only when gradients are synchronized
        if accelerator.sync_gradients and scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # Compute metrics
        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)
            batch_metrics = unwrapped_model.compute_metrics(batch, outputs)

            for k, v in batch_metrics.items():
                if k not in train_metrics:
                    train_metrics[k] = AverageMeter(k, ":.4f")
                train_metrics[k].update(v, batch["batch_size"])

        train_losses.update(loss.item(), batch["batch_size"])

        # batch-level logging
        if (iteration % 5 == 0 or iteration == total_iterations - 1):
            # ----- eta computation -----
            elapsed   = time.time() - start_time
            done_frac = (iteration + 1) / total_iterations
            if done_frac > 0:                            # avoid div-by-zero
                remaining = elapsed * (1 - done_frac) / done_frac
            else:
                remaining = 0
            days  = int(remaining // 86400); remaining %= 86400
            hrs   = int(remaining // 3600);  remaining %= 3600
            mins  = int(remaining // 60);    secs = int(remaining % 60)
            eta_str = f"{days}d {hrs}h {mins}m {secs}s"
            # ---------------------------

            lr = optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
            local_bs = len(batch["left_image"])
            global_bs = local_bs * accelerator.num_processes
            accelerator.print(
                f"[{current_epoch+1}][{iteration+1}/{total_iterations}]"
                f"[{dataset_position}/{batches_per_epoch}] "
                f"train loss: {loss.item():.10f} | lr: {lr:.10f} | "
                f"grad norm: {grad_norm} | batch size: {local_bs} "
                f"(global: {global_bs}) | eta: {eta_str}"
            )

        # Memory logging (every 100 iterations)
        if iteration % 100 == 0 and accelerator.is_local_main_process and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            accelerator.print(f"[Memory] GPU {accelerator.local_process_index}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # wandb logging - only on main process
        if iteration % config.logging.wandb_interval == 0 and accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]["lr"]
            wandb_logger.log_training_batch(iteration, current_epoch, loss.item(), lr, loss_details, train_metrics, config, accelerator)

        
        # Save training visualizations - only on main process
        if iteration % 10 == 0 and accelerator.is_local_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            base_name = f"t_e_{current_epoch}_b_{iteration}"
            unwrapped_model.save_visualizations(batch, outputs, base_name)

        # Validation and epoch-level summaries (every validation_frequency iterations)
        if (iteration + 1) % validation_frequency == 0 or iteration == total_iterations - 1:
            # epoch-level logging to wandb - only on main process
            if accelerator.is_local_main_process:
                wandb_logger.log_training_epoch(iteration, current_epoch, train_losses, train_metrics, optimizer, config, accelerator)

            # epoch summary for training
            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] train epoch loss: {train_losses.avg:.10f}")

            train_metrics_str = ""
            for k, meter in train_metrics.items():
                train_metrics_str += f" | {k}: {meter.avg:.3f}"

            if train_metrics_str:
                accelerator.print(f"[{current_epoch+1}] train{train_metrics_str}")

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
                    val_outputs = model(val_batch)
                    
                    # Loss computation in FP32 to prevent NaN with ConfLoss  
                    unwrapped_model = accelerator.unwrap_model(model)
                    val_loss, val_loss_details = unwrapped_model.get_loss(criterion, val_batch, val_outputs)

                    # Handle potential None loss in validation
                    if val_loss is None or not isinstance(val_loss, torch.Tensor):
                        val_loss = torch.zeros(1, device=accelerator.device)
                        accelerator.print(f"[Validation] Got zero/null loss at batch {val_batch_idx}")

                    # compute metrics
                    val_batch_metrics = unwrapped_model.compute_metrics(val_batch, val_outputs)

                    for k, v in val_batch_metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = AverageMeter(k, ":.4f")
                        val_metrics[k].update(v, val_batch["batch_size"])

                    val_losses.update(val_loss.item(), val_batch["batch_size"])

                    # Add validation batch logging here - only on main process
                    if val_batch_idx % config.logging.wandb_interval == 0 and accelerator.is_local_main_process:
                        wandb_logger.log_validation_batch(
                            iteration, current_epoch, val_batch_idx, val_loss.item(), 
                            val_loss_details, val_batch_metrics, config, accelerator
                        )

                    # logging
                    if (val_batch_idx == 0 or (val_batch_idx + 1) % 5 == 0 or val_batch_idx == len(valid_loader) - 1):
                        accelerator.print(
                            f"[{current_epoch+1}][{iteration+1}/{total_iterations}][{val_batch_idx+1}/{len(valid_loader)}] valid loss: {val_loss.item():.10f}"
                        )

                # Create valid directory and save visualizations only on main process
                if accelerator.is_local_main_process:
                    valid_vis_dir = os.path.join(output_dir, "valid")
                    os.makedirs(valid_vis_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    base_name = f"v_e_{current_epoch}_b_{iteration}"
                    unwrapped_model.save_visualizations(val_batch, val_outputs, base_name)

            # wandb logging for validation - only on main process
            if accelerator.is_local_main_process:
                wandb_logger.log_validation_epoch(iteration, current_epoch, val_losses, val_metrics, config, accelerator)

            # epoch summary for validation
            accelerator.print(f"[{current_epoch+1}][{iteration+1}/{total_iterations}] val epoch loss: {val_losses.avg:.10f}")

            val_metrics_str = ""
            for k, meter in val_metrics.items():
                val_metrics_str += f" | {k}: {meter.avg:.3f}"

            if val_metrics_str:
                accelerator.print(f"[{current_epoch+1}] val{val_metrics_str}")

            # save best model - only on main process
            if config.valid.save.save_best and accelerator.is_local_main_process:
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
                    metric_val = None

                if metric_val is not None:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if not hasattr(unwrapped_model, "best_metric"):
                        unwrapped_model.best_metric = float("inf") if better(0, 1) else float("-inf")
                        is_best = True
                    elif better(metric_val, unwrapped_model.best_metric):
                        is_best = True
                        unwrapped_model.best_metric = metric_val

                    if is_best:
                        # Ensure all processes wait before saving
                        accelerator.wait_for_everyone()
                        checkpoint_path = save_best_model(
                            accelerator, model, optimizer, scheduler, iteration, current_epoch,
                            unwrapped_model.best_metric, metric_name, val_losses.avg, config, output_dir
                        )

            # Reset training metrics for next epoch period
            train_losses.reset()
            for k, meter in train_metrics.items():
                meter.reset()

            model.train()
        
        # Clear GPU cache periodically to prevent memory issues
        if iteration % 1000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # --- profiler & memlab per-iter hooks ---------------------------
        if profiler:
            profiler.step()                          # advance profiler state

        if config.debug and HAS_MEMLAB and iteration % 100 == 0 and accelerator.is_local_main_process:
            MemReporter().report()                  # light live report
            if torch.cuda.is_available():           # start fresh for the next window
                torch.cuda.reset_peak_memory_stats()
        # ----------------------------------------------------------------

        # Increment iteration counter
        iteration += 1

    # stop profiler
    if profiler:
        profiler.stop()
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()
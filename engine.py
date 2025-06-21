import torch
import torch.nn.functional as F
import wandb
import numpy as np
import time

from hydra.core.hydra_config import HydraConfig


def train_one_epoch(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    accelerator,
    epoch,
    config,
):
    """
    Train the model for one complete epoch.

    This function handles the complete training cycle for one epoch, including forward and
    backward passes, optimization steps, metric tracking, logging, and optional validation.
    It uses the accelerator for distributed training support.

    Parameters:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader providing the training data batches
        valid_loader (DataLoader): DataLoader providing validation data for interim validation
        criterion (callable): Loss function that takes (batch, outputs) and returns (loss, loss_details)
        optimizer (Optimizer): PyTorch optimizer for model parameter updates
        scheduler (LRScheduler): Learning rate scheduler to adjust LR during training
        accelerator (Accelerator): HuggingFace Accelerator for distributed training
        epoch (int): Current epoch number (0-indexed)
        config (dict/object): Configuration object containing training parameters

    Returns:
        dict: Dictionary of training results containing 'loss' and any other tracked metrics
             averaged over the entire epoch
    """
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    metrics = {}

    for batch_idx, batch in enumerate(train_loader):
        # forward prop
        with accelerator.autocast():
            outputs = model(batch)
            # print("----- batch -----")
            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"{k}: {v.shape}")
            #     else:
            #         print(f"{k}: {v}")
            # print("----- outputs -----")
            # for k, v in outputs.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"{k}: {v.shape}")
            #     else:
            #         print(f"{k}: {v}")
            # print("----------------")
            # loss, loss_details = criterion(batch, outputs)
            loss, loss_details = model.get_loss(criterion, batch, outputs)

        # backward prop
        if isinstance(loss, torch.Tensor):
            accelerator.backward(loss)
            optimizer.step()
        else:
            print(f"train_one_epoch | GOT ZERO LOSS; which means the valid mask is not valid anywhere, check mask sum below:")
            for i in range(batch["left_image"].size(0)):
                print(f'get_loss | {i} | instance1: {batch["left_instance"][i]} | instance2: {batch["right_instance"][i]}')
                print(f'get_loss | {i} | mask1 sum: {batch["left_pm"][i][..., 3].sum()} | mask2 sum: {batch["right_pm"][i][..., 3].sum()} | og shape: {batch["left_pm"][i].shape}')

            loss = torch.zeros(1).to(batch["left_image"].device)

        # grad clip
        if config.train.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)

        # update weights
        optimizer.step()
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        # compute metrics
        with torch.no_grad():
            batch_metrics = model.compute_metrics(batch, outputs)

            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = AverageMeter(k, ":.4f")
                metrics[k].update(v, batch["batch_size"])

        losses.update(loss.item(), batch["batch_size"])

        # viz logging
        if (
            batch_idx == 0
            or (batch_idx + 1) % 5 == 0
            or batch_idx == len(train_loader) - 1
        ):
            lr = (
                optimizer.param_groups[0]["lr"]
                if scheduler is None
                else scheduler.get_last_lr()[0]
            )
            accelerator.print(
                f"[{epoch+1}/{config.train.epochs}][{batch_idx+1}/{len(train_loader)}] train loss: {loss.item():.10f} | lr: {lr:.10f} | grad norm: {grad_norm} |",
                end="",
            )
            for k, v in loss_details.items():
                accelerator.print(f" {k}: {v:.4f}", end=" |")
            accelerator.print("")
            
        if epoch % 10 == 0:
            model.save_visualizations(batch, outputs, epoch, batch_idx, HydraConfig.get().runtime.output_dir)

        # wandb logging
        if batch_idx % config.logging.wandb_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "epoch": epoch,
                "train_batch_idx": batch_idx,
                "train_batch_loss": loss.item(),
                "train_learning_rate": lr,
            }

            for k, v in loss_details.items():
                log_dict[f"train_{k}"] = v

            for k, meter in metrics.items():
                log_dict[f"train_batch_{k}"] = meter.val

            if accelerator.is_local_main_process and config.logging.use_wandb:
                wandb.log(log_dict)

        # validation
        if (
            config.train.val_interval > 0
            and (batch_idx + 1) % config.train.val_interval == 0
        ):
            val_metrics = valid_one_epoch(
                model,
                valid_loader,
                criterion,
                accelerator,
                epoch,
                config,
                optimizer,
                scheduler,
                prefix="val_interim",
            )
            model.train()

    # more logging
    if accelerator.is_local_main_process and config.logging.use_wandb:
        log_dict = {
            "epoch": epoch,
            "train_epoch_loss": losses.avg,
            "train_epoch_learning_rate": optimizer.param_groups[0]["lr"],
        }

        for k, meter in metrics.items():
            log_dict[f"train_epoch_{k}"] = meter.avg

        wandb.log(log_dict)

    # epoch summary
    accelerator.print(
        f"[{epoch+1}/{config.train.epochs}] train epoch loss: {losses.avg:.10f}"
    )

    metrics_str = ""
    for k, meter in metrics.items():
        metrics_str += f" | {k}: {meter.avg:.3f}"

    if metrics_str:
        accelerator.print(f"[{epoch+1}/{config.train.epochs}] train{metrics_str}")

    # lr scheduler step (for ReduceLROnPlateau)
    if scheduler is not None and isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        metric_name = config.sched.metric
        if metric_name == "loss":
            scheduler.step(losses.avg)
        elif metric_name in metrics:
            scheduler.step(metrics[metric_name].avg)

    # Collect results
    results = {"loss": losses.avg}
    for k, meter in metrics.items():
        results[k] = meter.avg

    return results


def valid_one_epoch(
    model,
    valid_loader,
    criterion,
    accelerator,
    epoch,
    config,
    optimizer=None,
    scheduler=None,
    prefix="val",
):
    """
    Validate the model for one complete epoch.

    This function performs a full validation pass over the validation dataset,
    computing loss and metrics without updating model parameters. It supports
    distributed evaluation via accelerator and provides detailed logging.

    Parameters:
        model (nn.Module): The neural network model to validate
        valid_loader (DataLoader): DataLoader providing the validation data
        criterion (callable): Loss function that takes (batch, outputs) and returns (loss, loss_details)
        accelerator (Accelerator): HuggingFace Accelerator for distributed evaluation
        epoch (int): Current epoch number (0-indexed)
        config (dict/object): Configuration object containing validation parameters
        optimizer (Optimizer, optional): PyTorch optimizer, needed for checkpointing
        scheduler (LRScheduler, optional): Learning rate scheduler, needed for checkpointing
        prefix (str): Prefix for metric logging, typically 'val' or 'test'

    Returns:
        dict: Dictionary of validation results containing 'loss' and all other tracked
             metrics averaged over the entire validation set
    """
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    metrics = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            # forward prop
            with accelerator.autocast():
                outputs = model(batch)
                loss, loss_details = model.get_loss(criterion, batch, outputs)

            # compute metrics
            batch_metrics = model.compute_metrics(batch, outputs)

            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = AverageMeter(k, ":.4f")
                metrics[k].update(v, batch["batch_size"])

            losses.update(loss.item(), batch["batch_size"])

            # logging
            if (
                batch_idx == 0
                or (batch_idx + 1) % 5 == 0
                or batch_idx == len(valid_loader) - 1
            ):
                accelerator.print(
                    f"[{epoch+1}/{config.train.epochs}][{batch_idx+1}/{len(valid_loader)}] valid loss: {loss.item():.10f}"
                )

    # wandb logging
    if accelerator.is_local_main_process and config.logging.use_wandb:
        log_dict = {
            "epoch": epoch,
            f"{prefix}_epoch_loss": losses.avg,
        }

        for k, meter in metrics.items():
            log_dict[f"{prefix}_epoch_{k}"] = meter.avg

        wandb.log(log_dict)

    # epoch summary
    accelerator.print(
        f"[{epoch+1}/{config.train.epochs}] {prefix} epoch loss: {losses.avg:.10f}"
    )

    metrics_str = ""
    for k, meter in metrics.items():
        metrics_str += f" | {k}: {meter.avg:.3f}"

    if metrics_str:
        accelerator.print(f"[{epoch+1}/{config.train.epochs}] {prefix}{metrics_str}")

    # save best model
    if accelerator.is_local_main_process and config.valid.save.save_best:
        is_best = False
        metric_name = config.valid.save.best_metric

        if metric_name == "loss":
            metric_val = losses.avg
            better = lambda new, old: new < old
        elif metric_name in metrics:
            metric_val = metrics[metric_name].avg
            better = lambda new, old: (
                new < old if config.valid.save.lower_is_better else new > old
            )
        else:
            accelerator.print(
                f"warning: best_metric '{metric_name}' not found in metrics."
            )
            return {
                "loss": losses.avg,
                **{k: meter.avg for k, meter in metrics.items()},
            }

        if not hasattr(model, "best_metric"):
            model.best_metric = float("inf") if better(0, 1) else float("-inf")
            is_best = True
        elif better(metric_val, model.best_metric):
            is_best = True
            model.best_metric = metric_val

        if is_best:
            model.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "best_metric": model.best_metric,
                    "optimizer": (
                        optimizer.state_dict() if optimizer is not None else None
                    ),
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                },
                is_best=True,
                filename=f"{config.valid.save.output_dir}/checkpoint_{epoch+1}.pth",
            )
            print(
                f"saved best model at path {config.valid.save.output_dir}/checkpoint_{epoch+1}.pth - epoch: {epoch+1} | loss: {losses.avg:.4f} | {metric_name}: {model.best_metric:.4f} "
            )

    # collect results
    results = {"loss": losses.avg}
    for k, meter in metrics.items():
        results[k] = meter.avg

    return results


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.

    This class provides a simple way to track running averages for any metric
    during training or evaluation loops. It maintains both the current value
    and the running average over multiple updates.

    Attributes:
        name (str): Name of the metric being tracked
        fmt (str): String format for printing the metric
        val (float): Most recent value added
        avg (float): Running average of all values
        sum (float): Sum of all values
        count (int): Number of values added
    """

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
        """
        Update the meter with new value.

        Parameters:
            val (float): The new value to include in the average
            n (int): The weight of the new value (typically batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation of the meter."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

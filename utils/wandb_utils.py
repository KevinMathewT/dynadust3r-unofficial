# utils/wandb_utils.py
import wandb

# single global step counter shared by all phases
_global_step = 0


def _log(payload: dict):
    """internal helper to log with a strictly increasing step"""
    global _global_step
    wandb.log(payload, step=_global_step)
    _global_step += 1


def log_training_batch(
    iteration,
    current_epoch,
    loss_item,
    lr,
    loss_details,
    train_metrics,
    config,
    accelerator,
):
    """log every training batch"""
    if not (accelerator.is_local_main_process and config.logging.use_wandb):
        return

    log_dict = {
        "iteration": iteration,
        "epoch": current_epoch,
        "train_batch_loss": loss_item,
        "train_learning_rate": lr,
    }

    for k, v in loss_details.items():
        log_dict[f"train_{k}"] = v

    for k, meter in train_metrics.items():
        log_dict[f"train_batch_{k}"] = meter.val

    _log(log_dict)


def log_interim_validation(
    iteration, current_epoch, val_losses, val_metrics, config, accelerator
):
    """optional mid-epoch val logging"""
    if not (accelerator.is_local_main_process and config.logging.use_wandb):
        return

    log_dict = {
        "iteration": iteration,
        "epoch": current_epoch,
        "val_interim_loss": val_losses.avg,
    }

    for k, meter in val_metrics.items():
        log_dict[f"val_interim_{k}"] = meter.avg

    _log(log_dict)


def log_training_epoch(
    iteration,
    current_epoch,
    train_losses,
    train_metrics,
    optimizer,
    config,
    accelerator,
):
    """log aggregated train stats once per epoch window"""
    if not (accelerator.is_local_main_process and config.logging.use_wandb):
        return

    log_dict = {
        "iteration": iteration,
        "epoch": current_epoch,
        "train_epoch_loss": train_losses.avg,
        "train_epoch_learning_rate": optimizer.param_groups[0]["lr"],
    }

    for k, meter in train_metrics.items():
        log_dict[f"train_epoch_{k}"] = meter.avg

    _log(log_dict)


def log_validation_epoch(
    iteration, current_epoch, val_losses, val_metrics, config, accelerator
):
    """log aggregated validation stats once per epoch window"""
    if not (accelerator.is_local_main_process and config.logging.use_wandb):
        return

    log_dict = {
        "iteration": iteration,
        "epoch": current_epoch,
        "val_epoch_loss": val_losses.avg,
    }

    for k, meter in val_metrics.items():
        log_dict[f"val_epoch_{k}"] = meter.avg

    _log(log_dict)


def log_validation_batch(
    iteration,
    epoch,
    batch_idx,
    loss,
    loss_details,
    batch_metrics,
    config,
    accelerator,
):
    """log every validation batch"""
    if not (accelerator.is_local_main_process and config.logging.use_wandb):
        return

    log_dict = {
        "val_batch/loss": loss,
        "val_batch/epoch": epoch,
        "val_batch/iteration": iteration,
        "val_batch/batch_idx": batch_idx,
    }

    for k, v in loss_details.items():
        log_dict[f"val_batch/{k}"] = v

    for k, v in batch_metrics.items():
        log_dict[f"val_batch/{k}"] = v

    # pad train fields so charts line up
    log_dict["train_batch_loss"] = None
    log_dict["train_learning_rate"] = None

    _log(log_dict)

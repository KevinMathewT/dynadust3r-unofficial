import math
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR, LambdaLR
)
from omegaconf import OmegaConf


OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop
}


class CosineWarmupScheduler(LambdaLR):
    """
    Cosine decay scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epoch (int): Number of steps per epoch.
        epochs (int): Total number of training epochs.
        warmup_pct (float): Fraction of total steps used for linear warmup.
    """
    def __init__(self, optimizer, steps_per_epoch, epochs, warmup_pct=0.0):
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(warmup_pct * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))

        super().__init__(optimizer, lr_lambda)


class LinearWarmupScheduler(LambdaLR):
    """
    Linear decay scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epoch (int): Number of steps per epoch.
        epochs (int): Total number of training epochs.
        warmup_pct (float): Fraction of total steps used for linear warmup.
    """
    def __init__(self, optimizer, steps_per_epoch, epochs, warmup_pct=0.0):
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(warmup_pct * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

        super().__init__(optimizer, lr_lambda)


SCHEDULERS = {
    "steplr": StepLR,
    "exponentiallr": ExponentialLR,
    "reducelronplateau": ReduceLROnPlateau,
    "onecycle": OneCycleLR,
    "cosine": CosineWarmupScheduler,
    "linear": LinearWarmupScheduler
}


def get_optimizer(params, config):
    """
    Instantiates an optimizer using config.

    Args:
        params (iterable): Model parameters.
        config (DictConfig): Config containing `optim.name` and its args.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    opt_config = OmegaConf.to_container(config.optim, resolve=True)
    name = opt_config.pop("name")
    return OPTIMIZERS[name.lower()](params, **opt_config)


def get_scheduler(optimizer, config, loader):
    """
    Instantiates a scheduler using config.

    Args:
        optimizer (Optimizer): Optimizer for which to schedule learning rate.
        config (DictConfig): Config with `sched.name` and related args.
        loader (Iterable): Training data loader (used to get steps per epoch).

    Returns:
        _LRScheduler: Configured learning rate scheduler.
    """
    sched_config = OmegaConf.to_container(config.sched, resolve=True)
    name = sched_config.pop("name")
    sched_cls = SCHEDULERS.get(name)
    if sched_cls is None:
        raise ValueError(f"Unsupported scheduler: {name}")

    if name in ["cosine", "linear", "onecycle"]:
        return sched_cls(
            optimizer,
            steps_per_epoch=len(loader),
            epochs=config.train.epochs,
            **sched_config
        )
    return sched_cls(optimizer, **sched_config)

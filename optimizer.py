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
    cosine decay scheduler with warmup.

    args:
      optimizer: wrapped optimizer
      total_steps: total training steps
      warmup_pct: fraction of steps to warm up
    """
    def __init__(self, optimizer, total_steps, warmup_pct=0.0):
        warmup_steps = int(warmup_pct * total_steps)
        def lr_lambda(step):
            # linear warmup
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # cosine decay
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * p))
        super().__init__(optimizer, lr_lambda)  # no .step() dims here

class LinearWarmupScheduler(LambdaLR):
    """
    linear decay scheduler with warmup.

    args:
      optimizer: wrapped optimizer
      total_steps: total training steps
      warmup_pct: fraction of steps to warm up
    """
    def __init__(self, optimizer, total_steps, warmup_pct=0.0):
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
            total_steps=config.train.iterations,
            **sched_config
        )
    return sched_cls(optimizer, **sched_config)

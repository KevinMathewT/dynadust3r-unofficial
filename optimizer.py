import math

from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR


def get_optimizer(parameters, config):
    name = config.optim.name

    if name == "adam":
        return Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    if name == "adamw":
        return AdamW(parameters, lr=config.optim.lr, betas=tuple(config.optim.betas))
    elif name == "sgd":
        return SGD(parameters, lr=config.optim.lr, momentum=config.optim.momentum, weight_decay=config.optim.weight_decay)
    elif name == "rmsprop":
        return RMSprop(parameters, lr=config.optim.lr, momentum=config.optim.momentum, weight_decay=config.optim.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer, config, train_loader):
    name = config.sched.name
    epochs = config.train.epochs
    total_steps = len(train_loader) * epochs

    if name == "steplr":
        return StepLR(optimizer, step_size=config.sched.step_size, gamma=config.sched.gamma)
    elif name == "exponentiallr":
        return ExponentialLR(optimizer, gamma=config.sched.gamma)
    elif name == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, mode=config.sched.mode, factor=config.sched.factor, patience=config.sched.patience)
    elif name == "cosine":
        warmup_steps = int(config.sched.warmup_pct * total_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            cosine_progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1 + math.cos(math.pi * cosine_progress)))
        return LambdaLR(optimizer, lr_lambda)
    elif name == "onecycle":
        return OneCycleLR(optimizer, max_lr=config.sched.max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    elif name == "linear":
        warmup_steps = int(config.sched.warmup_pct * total_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    import torch

    optimizer = get_optimizer([torch.tensor([1.0, 2.0, 3.0])], config)
    scheduler = get_scheduler(optimizer, config, [1, 2, 3, 4, 5])

    print(optimizer)
    print(f"{config.sched.name} --> {scheduler.__class__.__name__}")

if __name__ == "__main__":
    main()
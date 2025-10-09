"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import math
import json
import os
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


def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith('decoder_embed') or var_name.startswith('enc_norm'): # part of the last black
        return enc_depth
    elif var_name.startswith('dec_blocks'):
        layer_id = int(var_name.split('.')[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith('dec_norm'): # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ['head','prediction_head']):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)


def get_parameter_groups(model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values
    assert layer_decay==1.0 or 0.<layer_decay<1.
    if layer_decay<1.:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, 'dec_blocks') else 0
        num_layers = enc_depth+dec_depth
        layer_decay_values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay<1.:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "names": []
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_vars[group_name]["names"].append(name)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_optimizer(model, config):
    """
    Instantiates an optimizer using config with parameter groups for layer decay.

    Args:
        model: Model with parameters to optimize.
        config (DictConfig): Config containing `optim.name` and its args.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    # Extract parameter group settings from config, with defaults
    weight_decay = getattr(config.optim, 'weight_decay', 0.01)
    layer_decay = getattr(config.optim, 'layer_decay', 1.0)  # 1.0 means no layer decay
    skip_list = getattr(config.optim, 'skip_list', ())
    no_lr_scale_list = getattr(config.optim, 'no_lr_scale_list', [])

    # Get parameter groups
    param_groups = get_parameter_groups(model, weight_decay, layer_decay, skip_list, no_lr_scale_list)

    opt_config = OmegaConf.to_container(config.optim, resolve=True)
    name = opt_config.pop("name")
    # Remove weight_decay from opt_config since it's handled in parameter groups
    opt_config.pop("weight_decay", None)
    opt_config.pop("layer_decay", None)
    opt_config.pop("skip_list", None)
    opt_config.pop("no_lr_scale_list", None)

    opt = OPTIMIZERS[name.lower()](param_groups, **opt_config)

    return opt


def get_scheduler(optimizer, config, loader):
    """
    Instantiates a scheduler using config.

    Args:
        optimizer (Optimizer): Optimizer for which to schedule learning rate.
        config (DictConfig): Config with `sched.name` and related args.
        loader (Iterable): Training data loader (used to get steps per epoch).

    Returns:
        _LRScheduler or None: Configured learning rate scheduler, or None if no scheduler.
    """
    # Handle case where scheduler is disabled (None)
    if config.sched is None:
        return None
    
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

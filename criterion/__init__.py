from omegaconf import OmegaConf
from hydra.utils import to_absolute_path

from criterion.criterion import (
    ConfLoss,
    Regr3D,
    Regr3D_ShiftInv,
    Regr3D_ScaleInv,
    Regr3D_ScaleShiftInv,
    L21Loss,
)


def _get_criterion(criterion):
    name = criterion.name

    if name == "conf":
        inner_path = to_absolute_path(f"config/criterion/{criterion.inner}.yaml")
        inner_config = OmegaConf.load(inner_path)
        inner = _get_criterion(inner_config)
        return ConfLoss(inner, alpha=criterion.alpha)

    loss = L21Loss(reduction=criterion.reduction)
    kw = dict(norm_mode=criterion.norm_mode, gt_scale=criterion.gt_scale)

    if name == "regr3d":
        return Regr3D(loss, **kw)
    elif name == "regr3d_shiftinv":
        return Regr3D_ShiftInv(loss, **kw)
    elif name == "regr3d_scaleinv":
        return Regr3D_ScaleInv(loss, **kw)
    elif name == "regr3d_scaleshiftinv":
        return Regr3D_ScaleShiftInv(loss, **kw)

    raise ValueError(f"Unsupported criterion: {name}")


def get_criterion(config):
    return _get_criterion(config.criterion)

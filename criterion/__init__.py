from criterion.criterion import (
    ConfLoss,
    Regr3D,
    Regr3D_ShiftInv,
    Regr3D_ScaleInv,
    Regr3D_ScaleShiftInv,
    L21Loss,
)


def _get_criterion(cfg):
    if cfg.name == "conf":
        inner = get_criterion(cfg.inner)
        return ConfLoss(inner, alpha=cfg.alpha)

    if cfg.name == "regr3d":
        return Regr3D(
            L21Loss(reduction=cfg.reduction),
            norm_mode=cfg.norm_mode,
            gt_scale=cfg.gt_scale,
        )
    elif cfg.name == "regr3d_shiftinv":
        return Regr3D_ShiftInv(
            L21Loss(reduction=cfg.reduction),
            norm_mode=cfg.norm_mode,
            gt_scale=cfg.gt_scale,
        )
    elif cfg.name == "regr3d_scaleinv":
        return Regr3D_ScaleInv(
            L21Loss(reduction=cfg.reduction),
            norm_mode=cfg.norm_mode,
            gt_scale=cfg.gt_scale,
        )
    elif cfg.name == "regr3d_scaleshiftinv":
        return Regr3D_ScaleShiftInv(
            L21Loss(reduction=cfg.reduction),
            norm_mode=cfg.norm_mode,
            gt_scale=cfg.gt_scale,
        )

    raise ValueError(f"Unsupported criterion: {cfg.name}")


def get_criterion(config):
    return _get_criterion(config.criterion)

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory with motion head support
# --------------------------------------------------------
from models.dust3r.utils.heads.linear_head import LinearPts3d
from models.dust3r.utils.heads.dpt_head import create_dpt_head
from models.dust3r.utils.heads.motion_dpt_head import create_motion_dpt_head


def head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")


def motion_head_factory(head_type, output_mode, net, has_conf=False, time_pos_emb_dim=128):
    """" build a motion prediction head for the decoder 
    """
    if head_type == 'dpt' and output_mode == 'pts3d':
        return create_motion_dpt_head(net, has_conf=has_conf, time_pos_emb_dim=time_pos_emb_dim)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=} for motion head")
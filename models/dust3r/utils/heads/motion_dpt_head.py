import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import List
from models.dust3r.utils.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from croco.models.dpt_block import DPTOutputAdapter  # noqa

# taken from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py#L27
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    Create sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor): a 1-D Tensor of N indices, one per batch element. May be fractional.
        embedding_dim (int): dimension of the output.
        flip_sin_to_cos (bool): whether the embedding order should be cos, sin instead of sin, cos.
        downscale_freq_shift (float): controls the delta between frequencies across dimensions.
        scale (float): scaling factor applied to embeddings.
        max_period (int): controls the maximum frequency of the embeddings.

    Returns:
        torch.Tensor: shape [N, embedding_dim], the sinusoidal positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor"  # [N]

    half_dim = embedding_dim // 2  # scalar
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )  # [half_dim]
    exponent = exponent / (half_dim - downscale_freq_shift)  # [half_dim]

    emb = torch.exp(exponent)  # [half_dim]
    emb = timesteps[:, None].float() * emb[None, :]  # [N, half_dim]

    emb = scale * emb  # [N, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [N, 2*half_dim]

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)  # [N, 2*half_dim]

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # [N, embedding_dim]

    return emb  # [N, embedding_dim]


class MotionDPTOutputAdapter(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weights, and fix forward for dust3r
    with added support for time query
    """
    def __init__(self, time_pos_emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_pos_emb_dim = time_pos_emb_dim
        self.linear_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_pos_emb_dim, out_dim),  # [B, out_dim]
                nn.SiLU(),
                nn.Linear(out_dim, out_dim)  # [B, out_dim]
            ) for out_dim in self.layer_dims
        ])

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        del self.act_1_postprocess  # remove duplicate weights
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, t=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        image_size = self.image_size if image_size is None else image_size  # (H, W)
        H, W = image_size
        N_H = H // (self.stride_level * self.P_H)  # scalar
        N_W = W // (self.stride_level * self.P_W)  # scalar

        layers = [encoder_tokens[hook] for hook in self.hooks]  # list of [B, N, C]
        layers = [self.adapt_tokens(l) for l in layers]  # list of [B, N, C]
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]  # list of [B, C, N_H, N_W]
        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]  # list of [B, layer_dims[i], N_H, N_W]

        time_emb = get_timestep_embedding(t, self.time_pos_emb_dim, downscale_freq_shift=0, flip_sin_to_cos=True)  # [B, emb_dim]

        layers = [self.scratch.layer_rn[idx](l) + self.linear_projections[idx](time_emb).unsqueeze(-1).unsqueeze(-1)
                  for idx, l in enumerate(layers)]  # list of [B, layer_dims[i], N_H, N_W]

        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)  # [B, num_channels, H, W]
        return out  # [B, num_channels, H, W]


class PixelwiseTaskWithMotionDPT(nn.Module):
    """ DPT module for dust3r motion prediction, can return 3D motion + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=3, postprocess=None, depth_mode=None, conf_mode=None, 
                 time_pos_emb_dim=128, **kwargs):
        super(PixelwiseTaskWithMotionDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.time_pos_emb_dim = time_pos_emb_dim

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        time_pos_emb_dim=time_pos_emb_dim,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = MotionDPTOutputAdapter(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info, query_time):
        # Process query_time to ensure it's a batch of 1D tensors
        if not isinstance(query_time, torch.Tensor):
            query_time = torch.tensor(query_time, device=x[0].device)
        if query_time.dim() == 0:
            query_time = query_time.unsqueeze(0)
        if query_time.dim() == 1 and len(query_time) == 1 and x[0].shape[0] > 1:
            query_time = query_time.expand(x[0].shape[0])
        
        out = self.dpt(x, image_size=(img_info[0], img_info[1]), t=query_time)
        
        # Create a dictionary with motion vectors
        result = {'motion': out}
        
        # Add confidence if needed
        if self.postprocess and out.shape[1] > 3:
            motion, conf = torch.split(out, [3, out.shape[1]-3], dim=1)
            result['motion'] = motion
            result['conf'] = conf
            
        return result


def create_motion_dpt_head(net, has_conf=False, time_pos_emb_dim=128):
    """
    return PixelwiseTaskWithMotionDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3  # xyz motion vectors
    if has_conf:
        out_nchan += 1  # add confidence channel
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithMotionDPT(num_channels=out_nchan,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=None,  # Handle postprocessing in forward
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                time_pos_emb_dim=time_pos_emb_dim,
                                head_type='regression')
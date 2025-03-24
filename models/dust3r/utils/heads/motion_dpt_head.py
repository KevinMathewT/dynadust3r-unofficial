import torch
import torch.nn as nn
from einops import rearrange
from typing import List
from croco.models.dpt_block import DPTOutputAdapter  # noqa

class TimePosEmbedding(nn.Module):
    """
    Computes sinusoidal time positional embeddings for a scalar t in [0,1].
    Output is of shape [B, emb_dim], where emb_dim is typically 128.
    """

    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        b = t.shape[0]  # [B]
        device = t.device
        dtype = t.dtype
        i = torch.arange(self.emb_dim, device=device, dtype=dtype)  # [emb_dim]
        angle_rates = 1.0 / (10000.0 ** ((2.0 * (i // 2)) / float(self.emb_dim)))  # [emb_dim]
        angle_rads = t.unsqueeze(1) * angle_rates.unsqueeze(0)  # [B, emb_dim]
        angle_rads[:, 0::2] = angle_rads[:, 0::2].sin()  # [B, emb_dim]
        angle_rads[:, 1::2] = angle_rads[:, 1::2].cos()  # [B, emb_dim]
        return angle_rads  # [B, emb_dim]

class MotionDPTOutputAdapter(DPTOutputAdapter):
    """
    Motion-aware DPTOutputAdapter that injects time conditioning into each layer
    using sinusoidal embeddings + MLP, following the DynaDUSt3R design.
    """

    def __init__(self, time_pos_emb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_pos_emb = time_pos_emb
        self.linear_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.time_pos_emb.emb_dim, out_dim),  # [B, out_dim]
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

        time_emb = self.time_pos_emb(t)  # [B, emb_dim]

        layers = [self.scratch.layer_rn[idx](l) + self.linear_projections[idx](time_emb).unsqueeze(-1).unsqueeze(-1)
                  for idx, l in enumerate(layers)]  # list of [B, layer_dims[i], N_H, N_W]

        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)  # [B, num_channels, H, W]
        return out  # [B, num_channels, H, W]

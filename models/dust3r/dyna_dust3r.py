# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

# --------------------------------------------------------
# DynaDUSt3R model class - extends DUSt3R with motion prediction
# --------------------------------------------------------

# standard library
import gc
import io
import os
from copy import deepcopy
from time import time

# third-party
import cv2
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from packaging import version
import huggingface_hub
from PIL import Image

# local
from utils.geometry import normalize_pointcloud
import utils.geometry as geom
from utils.rerun_viz import visualize_image, visualize_pm, visualize_sequence_from_pms
from models.croco.croco import CroCoNet  # noqa: F401
from models.dust3r.utils.heads import head_factory, motion_head_factory
from models.dust3r.utils.heads.postprocess import reg_dense_depth
from models.dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from models.dust3r.utils.patch_embed import get_patch_embed


inf = float("inf")

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), (
    "Outdated huggingface_hub version, " "please reinstall requirements.txt"
)


class DynaDUSt3R(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d", "motion"],
):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry), and to predict motion for given time queries.
    """

    def __init__(
        self,
        output_mode="pts3d",
        motion_output_mode="pts3d",
        head_type="linear",
        motion_head_type="linear",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        motion_depth_mode=("linear", -inf, inf),
        motion_conf_mode=("exp", 1, inf),
        depth_post_mode=("linear", -inf, inf),
        motion_depth_post_mode=("linear", -inf, inf),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",  # PatchEmbedDust3R or ManyAR_PatchEmbed
        time_pos_emb_dim=128,
        teacher_forcing=False,
        # Whether to use ground truth (True) or predicted (False) point clouds for motion
        # currently only False is supported - TODO: normalize/scale the gt point clouds correctly to match prediction space before adding predicted motion
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.time_pos_emb_dim = time_pos_emb_dim
        self.teacher_forcing = teacher_forcing
        print(f"[DynaDUSt3R] teacher_forcing = {teacher_forcing}")
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        # postprocess modes for mapping head outputs in loss/metrics
        self.depth_post_mode = depth_post_mode
        self.motion_depth_post_mode = motion_depth_post_mode
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(
            output_mode,
            motion_output_mode,
            head_type,
            motion_head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            motion_depth_mode,
            motion_conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

        # random test stuff:
        self.k = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device="cpu", strict=False, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            ckpt = torch.load(pretrained_model_name_or_path, map_location=device, weights_only=False)
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            
            model_kwargs = {}
            if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
                cfg = ckpt["config"]
                if "model" in cfg and isinstance(cfg["model"], dict):
                    model_kwargs = dict(cfg["model"])
                    model_kwargs.pop("name", None)

            model = cls(**model_kwargs) if model_kwargs else cls()
            model.load_state_dict(state_dict, strict=strict)
            model.to(device)
            model.eval()
            return model
        else:
            try:
                model = super(DynaDUSt3R, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        motion_output_mode,
        head_type,
        motion_head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        motion_depth_mode,
        motion_conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.motion_output_mode = motion_output_mode
        self.head_type = head_type
        self.motion_head_type = motion_head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.motion_depth_mode = motion_depth_mode
        self.motion_conf_mode = motion_conf_mode
        # allocate point heads
        self.downstream_head1 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        self.downstream_head2 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        # magic wrapper
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )

        # allocate motion heads
        self.motion_head1 = motion_head_factory(
            motion_head_type,
            motion_output_mode,
            self,
            has_conf=bool(motion_conf_mode),
            time_pos_emb_dim=self.time_pos_emb_dim,
        )
        self.motion_head2 = motion_head_factory(
            motion_head_type,
            motion_output_mode,
            self,
            has_conf=bool(motion_conf_mode),
            time_pos_emb_dim=self.time_pos_emb_dim,
        )
        # magic wrapper
        self.mhead1 = transpose_to_landscape(
            self.motion_head1, activate=landscape_only)
        self.mhead2 = transpose_to_landscape(
            self.motion_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1, img2, shape1, shape2
            )

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        """
        Run dual decoder streams over encoded token sequences.

        Args:
            f1: (B, S, D_enc) - encoder tokens for left view
            pos1: (B, S, D_pos) or (B, S, 2) depending on CroCo settings
            f2: (B, S, D_enc) - encoder tokens for right view
            pos2: (B, S, D_pos) or (B, S, 2)

        Returns:
            iterator over tuples (dec_left_stage, dec_right_stage), where each is (B, S, D_dec)
            across stages including the pre-projection stage and all decoder block stages.
            When consumed as: `dec_left, dec_right = self._decoder(...)`, each of
            `dec_left` and `dec_right` is a tuple of T tensors, each tensor (B, S, D_dec),
            where T = 1 (pre-proj) + dec_depth.
        """
        final_output = [(f1, f2)]  # [(B, S, D_enc), (B, S, D_enc)] before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)  # (B, S, D_dec)
        f2 = self.decoder_embed(f2)  # (B, S, D_dec)

        final_output.append((f1, f2))  # now: [((B, S, D_enc) x 2), ((B, S, D_dec) x 2)]
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)  # inputs: (B, S, D_dec),(B, S, D_dec); outputs f1: (B, S, D_dec)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)  # inputs: (B, S, D_dec),(B, S, D_dec); outputs f2: (B, S, D_dec)
            # store the result
            final_output.append((f1, f2))  # append stage outputs; tensors (B, S, D_dec)

        # normalize last output
        # At this point final_output is: [((B, S, D_enc) x 2), ((B, S, D_dec) x 2), dec_depth x ((B, S, D_dec) x 2)]
        # Remove the projected stage to keep: [((B, S, D_enc) x 2), dec_depth x ((B, S, D_dec) x 2)]
        del final_output[1]  # remove projected duplicate stage (keep pre-proj and all block stages)
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))  # apply norm: (B, S, D_dec)
        return zip(*final_output)  # [(B, S, D_enc), dec_depth x (B, S, D_dec)] x 2

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def _motion_head(self, head_num, decout, img_shape, query_time):
        B, S, D = decout[-1].shape
        head = getattr(self, f"mhead{head_num}")
        return head(decout, img_shape, query_time)

    def _motion_head_multi(self, head_num, decout, img_shape, query_times):
        """
        Process multiple query times in parallel for efficient motion prediction.
        
        Args:
            head_num: Which motion head to use (1 or 2)
            decout: tuple/list length = 1 + dec_depth; first (B, S, D_enc), rest (B, S, D_dec)
            img_shape: (B, 2) tensor of (H, W)
            query_times: (B, T) tensor where T is number of query times
            
        Returns:
            dict: { 'map_pred': (B, T, H, W, 3), optional 'map_pred_conf': (B, T, H, W, 1) }
        """
        B = decout[-1].shape[0]  # (int)
        T = query_times.shape[1]  # (int)
        
        # Prepare decoder outputs for batch processing
        # Replicate decoder outputs T times and flatten batch dimension
        decout_expanded = []
        for feat in decout:
            # feat: (B, S, D)
            feat_expanded = feat.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, S, D)
            feat_expanded = feat_expanded.reshape(B * T, *feat.shape[1:])  # (B*T, S, D)
            decout_expanded.append(feat_expanded)
        
        # Expand image shape for all time queries
        if isinstance(img_shape, torch.Tensor):
            img_shape_expanded = img_shape.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)  # (B*T, 2)
        else:
            # If img_shape is a tuple, repeat it
            img_shape_expanded = img_shape  # (H, W) tuple (not used in current flow)
        
        # Flatten query times for batch processing
        query_times_flat = query_times.reshape(B * T)  # (B*T,)
        
        # Get motion head and process all queries at once
        head = getattr(self, f"mhead{head_num}")
        motion_out = head(decout_expanded, img_shape_expanded, query_times_flat)
        # motion_out["map_pred"]: (B*T, H, W, 3); optional motion_out["map_pred_conf"]: (B*T, H, W)
        
        # Reshape outputs back to separate batch and time dimensions
        H, W = motion_out["map_pred"].shape[1:3]  # (int, int)
        motion_pred = motion_out["map_pred"].reshape(B, T, H, W, 3)  # (B, T, H, W, 3)
        
        result = {"map_pred": motion_pred}
        
        if "map_pred_conf" in motion_out:
            motion_conf = motion_out["map_pred_conf"].reshape(B, T, H, W, 1)  # (B, T, H, W, 1)
            result["map_pred_conf"] = motion_conf
        
        return result

    def forward(self, batch):
        """
        Forward pass for the DynaDUSt3R model with general multi-query time support.

        Parameters:
            batch (dict):
                - 'left_image': (B, 3, H, W)
                - 'right_image': (B, 3, H, W)
                - 'query_times': (B, T) or (T,) or None - arbitrary number of query times
                - 'left_instance': list[str] or None
                - 'right_instance': list[str] or None

        Returns:
            dict: Combined dictionary with results from both views including motion predictions
                - 'left_map_pred': (B, H, W, 3)
                - 'left_map_pred_conf': (B, H, W, 1) or None
                - 'right_map_pred_in_left_frame': (B, H, W, 3)
                - 'right_map_pred_conf': (B, H, W, 1) or None
                - 'motion_pred' (if query_times provided): dict mapping
                    - 'l_to_{t}': (B, H, W, 3) for available t
                    - 'r_to_{t}': (B, H, W, 3) for available t
                    - optional '..._conf': (B, H, W, 1)
                - 'batch_size': (int)
        """
        # extract views from batch
        left_view = {
            "img": batch["left_image"],  # (B, 3, H, W)
            "true_shape": torch.tensor(batch["left_image"].shape[-2:])[None].repeat(
                batch["left_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch.get("left_instance", None),  # (list[str]) or None
        }
        right_view = {
            "img": batch["right_image"],  # (B, 3, H, W)
            "true_shape": torch.tensor(batch["right_image"].shape[-2:])[None].repeat(
                batch["right_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch.get("right_instance", None),  # (list[str]) or None
        }

        # get query times
        query_times = batch.get("query_times", None)  # (B, T) or (T,) or None
        
        # encode images
        (shape_left, shape_right), (feat_left, feat_right), (pos_left, pos_right) = (
            self._encode_symmetrized(left_view, right_view)
        )  # shape_left: (B, 2), shape_right: (B, 2); feat_*: (B, S, D); pos_*: (B, S, D)

        # decode features
        dec_left, dec_right = self._decoder(feat_left, pos_left, feat_right, pos_right)  # dec_left, dec_right are tuples
        # each: [(B,S,D_enc), dec_depth x (B,S,D_dec)], length = 1 + dec_depth; together: 2 x [(B,S,D_enc), dec_depth x (B,S,D_dec)]

        with torch.amp.autocast(device_type=self.device_type, enabled=False):
            # get 3d points for both views
            res_left = self._downstream_head(
                1, [tok.float() for tok in dec_left], shape_left
            )  # decout: list len=1+dec_depth, first (B,S,D_enc), rest (B,S,D_dec); returns dict: map_pred: (B,H,W,3); optional map_pred_conf: (B,H,W)
            res_right = self._downstream_head(
                2, [tok.float() for tok in dec_right], shape_right
            )  # decout: list len=1+dec_depth, first (B,S,D_enc), rest (B,S,D_dec); returns dict: map_pred: (B,H,W,3); optional map_pred_conf: (B,H,W)

            # predict motion if time queries provided
            if query_times is not None:
                # Handle different query_times formats
                if not isinstance(query_times, torch.Tensor):
                    query_times = torch.tensor(query_times, device=batch["left_image"].device)
                
                # Ensure query_times has batch dimension
                if query_times.dim() == 1:
                    # If shape is (T,), expand to (B, T)
                    query_times = query_times.unsqueeze(0).expand(batch["left_image"].size(0), -1)
                
                # query_times shape: (B, T)
                B, T = query_times.shape  # (int, int)
                
                # Process all query times in parallel for both views
                motion_left_all = self._motion_head_multi(
                    1, [tok.float() for tok in dec_left], shape_left, query_times
                )  # decout: list len=1+dec_depth, first (B,S,D_enc), rest (B,S,D_dec); dict: map_pred: (B,T,H,W,3); optional map_pred_conf: (B,T,H,W,1)
                motion_right_all = self._motion_head_multi(
                    2, [tok.float() for tok in dec_right], shape_right, query_times
                )  # decout: list len=1+dec_depth, first (B,S,D_enc), rest (B,S,D_dec); dict: map_pred: (B,T,H,W,3); optional map_pred_conf: (B,T,H,W,1)
                
                # Organize motion predictions with standardized keys
                # Mid: l_to_t0, r_to_t0 (index-based)
                # Others: l_to_r (t==1), r_to_l (t==0)
                motion_pred = {}
                times = query_times[0]
                is_t0 = torch.isclose(times, torch.tensor(0.0, device=times.device), atol=1e-6, rtol=0.0)
                is_t1 = torch.isclose(times, torch.tensor(1.0, device=times.device), atol=1e-6, rtol=0.0)

                for t_idx in range(T):
                    # left-based: special case for t==1.0 → l_to_r, else index key if not t==0
                    if bool(is_t1[t_idx].item()):
                        key = "l_to_r"
                        motion_pred[key] = motion_left_all["map_pred"][:, t_idx]
                        if "map_pred_conf" in motion_left_all:
                            motion_pred[f"{key}_conf"] = motion_left_all["map_pred_conf"][:, t_idx]
                    elif not bool(is_t0[t_idx].item()):
                        key = f"l_to_t{t_idx}"
                        motion_pred[key] = motion_left_all["map_pred"][:, t_idx]
                        if "map_pred_conf" in motion_left_all:
                            motion_pred[f"{key}_conf"] = motion_left_all["map_pred_conf"][:, t_idx]

                    # right-based: special case for t==0.0 → r_to_l, else index key if not t==1
                    if bool(is_t0[t_idx].item()):
                        key = "r_to_l"
                        motion_pred[key] = motion_right_all["map_pred"][:, t_idx]
                        if "map_pred_conf" in motion_right_all:
                            motion_pred[f"{key}_conf"] = motion_right_all["map_pred_conf"][:, t_idx]
                    elif not bool(is_t1[t_idx].item()):
                        key = f"r_to_t{t_idx}"
                        motion_pred[key] = motion_right_all["map_pred"][:, t_idx]
                        if "map_pred_conf" in motion_right_all:
                            motion_pred[f"{key}_conf"] = motion_right_all["map_pred_conf"][:, t_idx]

        # Rename right's 3D points to indicate they're in left's frame
        res_right["map_pred_in_left_frame"] = res_right.pop("map_pred")  # (B, H, W, 3)

        if "map_pred_conf" in res_left and res_left["map_pred_conf"] is not None:
            if res_left["map_pred_conf"].ndim == 3:
                res_left["map_pred_conf"] = res_left["map_pred_conf"].unsqueeze(-1)  # (B, H, W, 1)
        if "map_pred_conf" in res_right and res_right["map_pred_conf"] is not None:
            if res_right["map_pred_conf"].ndim == 3:
                res_right["map_pred_conf"] = res_right["map_pred_conf"].unsqueeze(-1)  # (B, H, W, 1)


        # Combine results into single dictionary
        combined_results = {}  # dict[str, Tensor or dict]

        # Add left view 3D points
        for k, v in res_left.items():
            combined_results[f"left_{k}"] = v  # left_map_pred: (B, H, W, 3); left_map_pred_conf: (B, H, W, 1)

        # Add right view 3D points
        for k, v in res_right.items():
            combined_results[f"right_{k}"] = v  # right_map_pred_in_left_frame: (B, H, W, 3); right_map_pred_conf: (B, H, W, 1)

        # Add motion predictions if computed
        if query_times is not None:
            combined_results["motion_pred"] = motion_pred  # dict[str, Tensor]

        # Add batch size for metrics calculation
        combined_results["batch_size"] = batch["left_image"].size(0)  # (int)

        return combined_results  # dict
        # Output dictionary contains:
        # 'left_map_pred': (B, H, W, 3) - 3D points from left view
        # 'left_map_pred_conf': optional (B, H, W, 1) - confidence for left view points
        # 'right_map_pred_in_left_frame': (B, H, W, 3) - 3D points from right view in left frame
        # 'right_map_pred_conf': optional (B, H, W, 1) - confidence for right view points
        # 'motion_pred': dict of dynamic keys (present only if query_times is provided):
        #   - "l_to_{t}": (B, H, W, 3) motion from left (t=0) to time t for each queried t with t != 0
        #       - optional "l_to_{t}_conf": (B, H, W, 1) if confidence is enabled (conf_mode != None)
        #   - "r_to_{t}": (B, H, W, 3) motion from right (t=1) to time t for each queried t with t != 1
        #       - optional "r_to_{t}_conf": (B, H, W, 1) if confidence is enabled (conf_mode != None)
        #   - t is a float value taken from query_times[0, t_idx]; keys exist only for those specific t values
        # 'batch_size': integer - batch size for metrics calculation

    def get_loss(self, batch, outputs):
        """
        Compute total loss with aggressive memory optimization.
        Fixed to match unoptimized version behavior exactly.

        Args:
            batch (dict):
                - left_pm: (B, H, W, 4)
                - right_pm: (B, H, W, 4)
                - motion_gt (dict):
                    - 'l2m': (B, H, W, 4)
                    - 'r2m': (B, H, W, 4)
                    - 'l2r': (B, H, W, 4)
                    - 'r2l': (B, H, W, 4)
                - left_image: (B, 3, H, W)
                - right_image: (B, 3, H, W)
                - query_times: (B, T) or (T,)
            outputs (dict):
                - left_map_pred: (B, H, W, 3)
                - left_map_pred_conf: (B, H, W, 1) or None
                - right_map_pred_in_left_frame: (B, H, W, 3)
                - right_map_pred_conf: (B, H, W, 1) or None
                - motion_pred (dict):
                    - "l_to_{t}": (B, H, W, 3) for available t
                    - "r_to_{t}": (B, H, W, 3) for available t
                    - optional "..._conf": (B, H, W, 1)

        Returns:
            total_loss: ()
            loss_details: dict[str, float]
        """
        device = batch["left_pm"].device  # (torch.device)
        alpha = 0.2  # (float)

        # ------------------------------------------------------------------
        # Apply post-depth mappings BEFORE anything else, as requested
        # - depth_post_mode on raw point-head outputs
        # - motion_depth_post_mode on (point + motion) sums
        # ------------------------------------------------------------------
        # Base predictions (B, H, W, 3)
        left_pred_pp = reg_dense_depth(outputs["left_map_pred"], self.depth_post_mode)
        right_pred_pp = reg_dense_depth(outputs["right_map_pred_in_left_frame"], self.depth_post_mode)

        # Extract mid index (assumed at column 0) once
        tq_mid_idx = 0

        # Motion predictions (postprocessed sums)
        # Note: we use the same keys as in the current loss config
        # and only access those keys that are used below
        l2m_key = f"l_to_t{tq_mid_idx}"
        r2m_key = f"r_to_t{tq_mid_idx}"
        l2r_key = "l_to_r"
        r2l_key = "r_to_l"

        # Helper to postprocess motion+point sums
        def _pp_motion_sum(base_pc, motion_disp):
            return reg_dense_depth(base_pc + motion_disp[..., :3], self.motion_depth_post_mode)

        # Build motion-postprocessed predictions
        pred_l2m_pp = _pp_motion_sum(outputs["left_map_pred"], outputs["motion_pred"][l2m_key])
        pred_r2m_pp = _pp_motion_sum(outputs["right_map_pred_in_left_frame"], outputs["motion_pred"][r2m_key])
        pred_l2r_pp = _pp_motion_sum(outputs["left_map_pred"], outputs["motion_pred"][l2r_key])
        pred_r2l_pp = _pp_motion_sum(outputs["right_map_pred_in_left_frame"], outputs["motion_pred"][r2l_key])

        # Process everything in chunks to avoid large intermediate tensors
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)  # ()
        loss_details = {}  # (dict)

        loss_configs = [
            {
                "name": "left",
                "gt": batch["left_pm"][..., :3],  # (B, H, W, 3)
                "pred": left_pred_pp,  # (B, H, W, 3) postprocessed
                "valid": batch["left_pm"][..., 3] > 0,  # (B, H, W)
                "conf": outputs.get("left_map_pred_conf", None),  # (B, H, W, 1) or None
                "is_base": True
            },
            {
                "name": "right", 
                "gt": batch["right_pm"][..., :3],  # (B, H, W, 3)
                "pred": right_pred_pp,  # (B, H, W, 3) postprocessed
                "valid": batch["right_pm"][..., 3] > 0,  # (B, H, W)
                "conf": outputs.get("right_map_pred_conf", None),  # (B, H, W, 1) or None
                "is_base": True
            },
            {
                "name": "l2m",
                "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2m"][..., :3],
                "pred": pred_l2m_pp,
                "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2m"][..., 3] > 0),
                "conf": outputs["motion_pred"].get(f"l_to_t{tq_mid_idx}_conf", None),
                "is_base": False
            },
            {
                "name": "r2m",
                "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2m"][..., :3],
                "pred": pred_r2m_pp,
                "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2m"][..., 3] > 0),
                "conf": outputs["motion_pred"].get(f"r_to_t{tq_mid_idx}_conf", None),
                "is_base": False
            },
            {
                "name": "l2r",
                "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2r"][..., :3],
                "pred": pred_l2r_pp,
                "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2r"][..., 3] > 0),
                "conf": outputs["motion_pred"].get("l_to_r_conf", None),
                "is_base": False
            },
            {
                "name": "r2l",
                "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2l"][..., :3],
                "pred": pred_r2l_pp,
                "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2l"][..., 3] > 0),
                "conf": outputs["motion_pred"].get("r_to_l_conf", None),
                "is_base": False
            }
        ]  # (list[len=6])

        # Compute normalization factors once for base PCs
        with torch.no_grad():  # ()
            # Extract base PCs and validity
            gt_left_pc = batch["left_pm"][..., :3]  # (B, H, W, 3)
            gt_right_pc = batch["right_pm"][..., :3]  # (B, H, W, 3)
            valid_left = batch["left_pm"][..., 3] > 0  # (B, H, W)
            valid_right = batch["right_pm"][..., 3] > 0  # (B, H, W)

            # GT normalization - using the FIXED version to match unoptimized
            gt_scale = self._compute_norm_factor_fixed(  # (B,1,1,1)
                gt_left_pc, gt_right_pc, valid_left, valid_right
            )

        # Pred normalization (needs gradients) — use postprocessed base PCs
        pred_scale = self._compute_norm_factor_fixed(  # (B,1,1,1)
            left_pred_pp,
            right_pred_pp,
            valid_left, valid_right
        )

        # Process each loss component
        for cfg in loss_configs:  # ()
            # # debug: print shapes per component once
            # if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            #     def s(x): return None if x is None else tuple(x.shape)
            #     print(f"[loss-cfg] {cfg['name']}: gt={s(cfg['gt'])} pred={s(cfg['pred'])} valid={s(cfg['valid'])} conf={s(cfg['conf'])}")
            #     if cfg["conf"] is not None and cfg["conf"].ndim == 4 and cfg["conf"].shape[-1] != 1:
            #         print(f"[conf-multi] {cfg['name']} conf has C={cfg['conf'].shape[-1]} channels")  # (B, H, W, C)

            # Compute single loss component
            loss_comp, comp_stats = self._compute_single_loss_fixed(  # ()
                cfg["gt"], cfg["pred"], cfg["valid"], cfg["conf"],
                gt_scale, pred_scale, alpha, device
            )

            # Accumulate loss
            total_loss = total_loss + loss_comp  # ()

            # Store details - FIXED to match unoptimized naming and extended stats
            if cfg["conf"] is not None:  # ()
                # (float)
                loss_details[f"{cfg['name']}_conf"] = loss_comp.item()
                # L2 not used when conf available  # (float)
                loss_details[f"{cfg['name']}_l2"] = 0.0
            else:
                loss_details[f"{cfg['name']}_conf"] = 0.0  # (float)
                loss_details[f"{cfg['name']}_l2"] = loss_comp.item()  # (float)

            # Extended statistics for L2 and confidence losses
            loss_details[f"{cfg['name']}_l2_mean"] = float(comp_stats.get("l2_mean", 0.0))
            loss_details[f"{cfg['name']}_l2_median"] = float(comp_stats.get("l2_median", 0.0))
            loss_details[f"{cfg['name']}_l2_var"] = float(comp_stats.get("l2_var", 0.0))
            loss_details[f"{cfg['name']}_conf_mean"] = float(comp_stats.get("conf_mean", 0.0))
            loss_details[f"{cfg['name']}_conf_median"] = float(comp_stats.get("conf_median", 0.0))
            loss_details[f"{cfg['name']}_conf_var"] = float(comp_stats.get("conf_var", 0.0))

            # Explicitly delete intermediate tensors if not base PC
            if not cfg["is_base"]:  # ()
                del cfg["gt"], cfg["pred"]  # ()
                if "valid" in cfg:  # ()
                    del cfg["valid"]  # ()

        loss_details["total_loss"] = total_loss.item()  # (float)

        # Force garbage collection for large tensors
        if hasattr(torch.cuda, 'empty_cache'):  # ()
            torch.cuda.empty_cache()  # ()

        return total_loss, loss_details  # ((), dict)

    def _compute_norm_factor_fixed(self, pc1, pc2, valid1, valid2):
        """
        Compute normalization factor matching unoptimized version exactly.
        Uses masked approach instead of extracting valid points.
        """
        batch_size = pc1.shape[0]  # ()
        device = pc1.device  # ()

        # Mask invalid points to 0 (matching unoptimized)
        pc1_masked = pc1.clone()  # (B, H, W, 3)
        pc2_masked = pc2.clone()  # (B, H, W, 3)
        pc1_masked[~valid1] = 0  # (B, H, W, 3)
        pc2_masked[~valid2] = 0  # (B, H, W, 3)

        # Stack and compute distances (matching unoptimized)
        all_pts = torch.cat(
            [pc1_masked.flatten(1, 2), pc2_masked.flatten(1, 2)], dim=1)  # (B,2*H*W,3)
        all_dis = all_pts.norm(dim=-1)  # (B,2*H*W)

        # Count valid points
        nnz1 = valid1.sum(dim=(1, 2))  # (B,)
        nnz2 = valid2.sum(dim=(1, 2))  # (B,)

        # Average distance
        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)  # (B,)
        norm_factor = norm_factor.clip(min=1e-8)  # (B,)

        # Expand to match PC dimensions
        while norm_factor.ndim < pc1.ndim:  # ()
            norm_factor = norm_factor.unsqueeze(-1)  # (B,1,1,1) after loop

        return norm_factor  # (B,1,1,1)

    def _compute_single_loss_fixed(self, gt_pc, pred_pc, valid_mask, conf, 
                                   gt_scale, pred_scale, alpha, device):
        """
        Compute loss for a single PC pair matching unoptimized behavior.
        Key fix: Only return confidence loss when available, otherwise return 0.
        """
        if not valid_mask.any():  # ()
            stats = {
                "l2_mean": 0.0,
                "l2_median": 0.0,
                "l2_var": 0.0,
                "conf_mean": 0.0,
                "conf_median": 0.0,
                "conf_var": 0.0,
            }
            return torch.tensor(0.0, device=pred_pc.device), stats  # ()
 
        # Normalize full PCs first (like unoptimized)
        gt_pc_norm = gt_pc / gt_scale  # (B, H, W, 3)
        pred_pc_norm = pred_pc / pred_scale  # (B, H, W, 3)
 
        # Compute L2 distance for all pixels
        l2_dist = (pred_pc_norm - gt_pc_norm).norm(dim=-1)  # (B, H, W)
 
        # Extract valid distances
        l2_dist_valid = l2_dist[valid_mask]  # (N,)

        # L2 statistics
        if l2_dist_valid.numel() > 0:
            l2_mean = l2_dist_valid.mean().item()
            l2_median = l2_dist_valid.median().item()
            l2_var = l2_dist_valid.var(unbiased=False).item()
        else:
            l2_mean = 0.0
            l2_median = 0.0
            l2_var = 0.0
 
        # CRITICAL: If confidence is provided, compute conf-weighted loss; otherwise use pure L2
        if conf is not None:  # ()
            # Extract valid confidence values
            conf_valid = conf[valid_mask].squeeze(-1)  # (N,)
            # Remove .squeeze(-1) unless you're sure about the shape
            # conf_valid = conf_valid.squeeze(-1)  
 
            # (K,), (K,)
            assert l2_dist_valid.ndim == 1 and conf_valid.ndim == 1
 
            # Compute confidence loss (no clamping to match unoptimized exactly)
            conf_loss_vec = (l2_dist_valid * conf_valid - alpha * torch.log(conf_valid))  # (N,)
            loss = conf_loss_vec.mean()  # ()
            if conf_loss_vec.numel() > 0:
                conf_mean = conf_loss_vec.mean().item()
                conf_median = conf_loss_vec.median().item()
                conf_var = conf_loss_vec.var(unbiased=False).item()
            else:
                conf_mean = 0.0
                conf_median = 0.0
                conf_var = 0.0
        else:
            # Pure L2 objective when confidence is disabled
            if l2_dist_valid.numel() > 0:
                loss = l2_dist_valid.mean()
            else:
                loss = torch.tensor(0.0, device=pred_pc.device)
            conf_mean = 0.0
            conf_median = 0.0
            conf_var = 0.0
 
        stats = {
            "l2_mean": l2_mean,
            "l2_median": l2_median,
            "l2_var": l2_var,
            "conf_mean": conf_mean,
            "conf_median": conf_median,
            "conf_var": conf_var,
        }

        return loss, stats  # ()

    # ---------------------------------------------------------------------
    #  UPDATED: compute_metrics – static error + *point-cloud* motion error
    # ---------------------------------------------------------------------

    def compute_metrics(self, batch, outputs):
        """
        Returns:
            dict: {metric_name: float}
        """
        metrics = {}

        # Precompute validity masks
        valid_left = batch["left_pm"][..., 3] > 0
        valid_right = batch["right_pm"][..., 3] > 0

        # Compute normalization factors exactly like in get_loss
        with torch.no_grad():
            gt_scale = self._compute_norm_factor_fixed(
                batch["left_pm"][..., :3],
                batch["right_pm"][..., :3],
                valid_left,
                valid_right,
            )

            pred_scale = None
            left_pred_pp = None
            right_pred_pp = None
            if "left_map_pred" in outputs and "right_map_pred_in_left_frame" in outputs:
                # Postprocess base predictions before computing pred_scale
                left_pred_pp = reg_dense_depth(outputs["left_map_pred"], self.depth_post_mode)
                right_pred_pp = reg_dense_depth(outputs["right_map_pred_in_left_frame"], self.depth_post_mode)
                pred_scale = self._compute_norm_factor_fixed(
                    left_pred_pp,
                    right_pred_pp,
                    valid_left,
                    valid_right,
                )

        # ================================================================
        #  1. static 3-D point-cloud errors (scale-normalized)
        # ================================================================
        if (
            pred_scale is not None
            and left_pred_pp is not None
            and valid_left.sum().item() > 0
        ):
            left_pred_n = left_pred_pp / pred_scale
            left_gt_n = batch["left_pm"][..., :3] / gt_scale
            metrics["left_3d_error"] = torch.norm(
                left_pred_n[valid_left] - left_gt_n[valid_left],
                dim=-1,
            ).mean().item()

        if (
            pred_scale is not None
            and right_pred_pp is not None
            and valid_right.sum().item() > 0
        ):
            right_pred_n = right_pred_pp / pred_scale
            right_gt_n = batch["right_pm"][..., :3] / gt_scale
            metrics["right_3d_error"] = torch.norm(
                right_pred_n[valid_right] - right_gt_n[valid_right],
                dim=-1,
            ).mean().item()

        if "left_3d_error" in metrics and "right_3d_error" in metrics:
            metrics["avg_3d_error"] = (
                metrics["left_3d_error"] + metrics["right_3d_error"]
            ) / 2.0

        # ================================================================
        #  2. motion errors – compare translated point-clouds (scale-normalized)
        # ================================================================
        if "motion_pred" not in outputs or "motion_gt" not in batch:
            return metrics  # nothing to add

        # Use index-based keys consistent with forward() and get_loss()
        tq_mid_idx = 0

        # Construct targets as base + GT motion (same as in loss computation)
        dir_cfg = {
            "l2m": {
                "pred_key": f"l_to_t{tq_mid_idx}",
                "src_pts_raw": outputs.get("left_map_pred"),
                "tgt_pts": batch["left_pm"][..., :3] + batch["motion_gt"]["l2m"][..., :3],
                "base_valid": valid_left,
                "motion_valid": batch["motion_gt"]["l2m"][..., 3] > 0,
            },
            "r2m": {
                "pred_key": f"r_to_t{tq_mid_idx}",
                "src_pts_raw": outputs.get("right_map_pred_in_left_frame"),
                "tgt_pts": batch["right_pm"][..., :3] + batch["motion_gt"]["r2m"][..., :3],
                "base_valid": valid_right,
                "motion_valid": batch["motion_gt"]["r2m"][..., 3] > 0,
            },
            "l2r": {
                "pred_key": "l_to_r",
                "src_pts_raw": outputs.get("left_map_pred"),
                "tgt_pts": batch["left_pm"][..., :3] + batch["motion_gt"]["l2r"][..., :3],
                "base_valid": valid_left,
                "motion_valid": batch["motion_gt"]["l2r"][..., 3] > 0,
            },
            "r2l": {
                "pred_key": "r_to_l",
                "src_pts_raw": outputs.get("right_map_pred_in_left_frame"),
                "tgt_pts": batch["right_pm"][..., :3] + batch["motion_gt"]["r2l"][..., :3],
                "base_valid": valid_right,
                "motion_valid": batch["motion_gt"]["r2l"][..., 3] > 0,
            },
        }

        motion_errs = []
        for name, cfg in dir_cfg.items():
            if cfg["pred_key"] not in outputs["motion_pred"]:
                continue  # prediction absent
            if cfg["src_pts_raw"] is None:
                continue  # base prediction absent

            # (B, H, W, 3)
            pred_disp = outputs["motion_pred"][cfg["pred_key"]]
            # sum raw base + motion and postprocess with motion_depth_post_mode
            pts_pred_pp = reg_dense_depth(cfg["src_pts_raw"] + pred_disp, self.motion_depth_post_mode)
            # intersection validity
            valid = cfg["base_valid"] & cfg["motion_valid"]

            if valid.sum().item() == 0:
                continue  # no valid GT for this direction

            # scale-normalized error (matching get_loss normalization)
            pts_pred_n = pts_pred_pp / pred_scale
            tgt_pts_n = cfg["tgt_pts"] / gt_scale

            err = torch.norm(
                pts_pred_n[valid] - tgt_pts_n[valid], dim=-1
            ).mean().item()

            metrics[f"{name}_motion_pc_error"] = err
            motion_errs.append(err)

        if motion_errs:
            metrics["avg_motion_pc_error"] = sum(
                motion_errs) / len(motion_errs)

        return metrics

    def save_visualizations(self, batch, outputs, base_name, i=0, *args, **kwargs):
        from utils.viz import save_visualizations as save_viz
        # pass depth post modes so viz applies same mapping as loss/metrics
        save_viz(
            batch,
            outputs,
            base_name,
            i=i,
            depth_post_mode=self.depth_post_mode,
            motion_depth_post_mode=self.motion_depth_post_mode,
            *args,
            **kwargs,
        )

    def save_checkpoint(self, state, is_best, filename):
        """
        Save model checkpoint.

        Parameters:
            state: Dictionary containing model state
            is_best: Whether this is the best model so far
            filename: Path to save the checkpoint
        """
        # save checkpoint
        torch.save(state, filename)

        # if this is the best model, save it as the best model
        if is_best:
            best_filename = filename.replace("checkpoint_", "model_best_")
            torch.save(state, best_filename)

    @staticmethod
    def load_model(cfg, device):
        """
        Load a DynaDUSt3R model from config, optionally downloading pretrained DUSt3R weights
        and copying point head weights to motion heads.

        Args:
            cfg (omegaconf.DictConfig): Config with model parameters.
            device (str): Device to map weights to.

        Returns:
            DynaDUSt3R: Initialized model instance.
        """
        import subprocess
        from omegaconf import OmegaConf

        # Extract model parameters from config
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg.pop("name", None)
        use_pretrained = model_cfg.pop("use_pretrained", False)
        pretrained_url = model_cfg.pop("pretrained_link", None)

        # Instantiate the model
        model = DynaDUSt3R(**model_cfg)

        if use_pretrained and pretrained_url is not None:
            print(f"using pretrained weights from {pretrained_url}")
            os.makedirs("weights/pretrained", exist_ok=True)
            filename = os.path.basename(pretrained_url)
            save_path = os.path.join("weights/pretrained", filename)

            # Download if not already present
            if not os.path.exists(save_path):
                print(f"downloading pretrained weights to {save_path}.")
                subprocess.run(
                    ["wget", pretrained_url, "-O", save_path], check=True)
            else:
                print(f"pretrained weights already exist at {save_path}.")

            # Load the checkpoint
            checkpoint = torch.load(
                save_path, map_location=device, weights_only=False)

            # Some checkpoints have 'model' and 'args' keys, so extract the raw state dict if needed
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                sd = checkpoint["model"]
            else:
                sd = checkpoint

            # Copy DUSt3R point-head keys into motion-head keys
            for k, v in list(sd.items()):
                if k.startswith("head1"):
                    sd[k.replace("head1", "mhead1")] = v.clone()
                elif k.startswith("head2"):
                    sd[k.replace("head2", "mhead2")] = v.clone()

            # Load into the DynaDUSt3R model
            model.load_state_dict(sd, strict=False)
            print(f"loaded pretrained weights successfully from {save_path}.")

        return model


if __name__ == "__main__":
    # test model instantiation
    model = DynaDUSt3R()
    print(model)

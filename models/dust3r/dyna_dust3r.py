# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
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


def load_model(model_path, device, verbose=True):
    pass


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
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(
            output_mode,
            motion_output_mode,
            head_type,
            motion_head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

        # random test stuff:
        self.k = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
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
            has_conf=bool(conf_mode),
            time_pos_emb_dim=self.time_pos_emb_dim,
        )
        self.motion_head2 = motion_head_factory(
            motion_head_type,
            motion_output_mode,
            self,
            has_conf=bool(conf_mode),
            time_pos_emb_dim=self.time_pos_emb_dim,
        )
        # magic wrapper
        self.mhead1 = transpose_to_landscape(self.motion_head1, activate=landscape_only)
        self.mhead2 = transpose_to_landscape(self.motion_head2, activate=landscape_only)

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
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

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
            decout: Decoder output features
            img_shape: Image shape information
            query_times: Tensor of shape (B, T) where T is number of query times
            
        Returns:
            dict: Motion predictions for all query times with shape (B, T, H, W, 3)
        """
        B = decout[-1].shape[0]
        T = query_times.shape[1]
        
        # Prepare decoder outputs for batch processing
        # Replicate decoder outputs T times and flatten batch dimension
        decout_expanded = []
        for feat in decout:
            # feat shape: (B, S, D)
            feat_expanded = feat.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, S, D)
            feat_expanded = feat_expanded.reshape(B * T, *feat.shape[1:])  # (B*T, S, D)
            decout_expanded.append(feat_expanded)
        
        # Expand image shape for all time queries
        if isinstance(img_shape, torch.Tensor):
            img_shape_expanded = img_shape.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)  # (B*T, 2)
        else:
            # If img_shape is a tuple, repeat it
            img_shape_expanded = img_shape
        
        # Flatten query times for batch processing
        query_times_flat = query_times.reshape(B * T)  # (B*T,)
        
        # Get motion head and process all queries at once
        head = getattr(self, f"mhead{head_num}")
        motion_out = head(decout_expanded, img_shape_expanded, query_times_flat)
        # motion_out["map_pred"] shape: (B*T, H, W, 3)
        
        # Reshape outputs back to separate batch and time dimensions
        H, W = motion_out["map_pred"].shape[1:3]
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
            batch: Dictionary containing:
                - 'left_image': (B, C, H, W)
                - 'right_image': (B, C, H, W)
                - 'query_times': (B, T) or (T,) - arbitrary number of query times

        Returns:
            dict: Combined dictionary with results from both views including motion predictions
        """
        # extract views from batch
        left_view = {
            "img": batch["left_image"],  # (B, C, H, W)
            "true_shape": torch.tensor(batch["left_image"].shape[-2:])[None].repeat(
                batch["left_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch.get("left_instance", None),  # [B x string] or None
        }
        right_view = {
            "img": batch["right_image"],  # (B, C, H, W)
            "true_shape": torch.tensor(batch["right_image"].shape[-2:])[None].repeat(
                batch["right_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch.get("right_instance", None),  # [B x string] or None
        }

        # get query times
        query_times = batch.get("query_times", None)  # (B, T) or (T,)
        
        # encode images
        (shape_left, shape_right), (feat_left, feat_right), (pos_left, pos_right) = (
            self._encode_symmetrized(left_view, right_view)
        )

        # decode features
        dec_left, dec_right = self._decoder(feat_left, pos_left, feat_right, pos_right)

        with torch.amp.autocast(device_type=self.device_type, enabled=False):
            # get 3d points for both views
            res_left = self._downstream_head(
                1, [tok.float() for tok in dec_left], shape_left
            )
            res_right = self._downstream_head(
                2, [tok.float() for tok in dec_right], shape_right
            )

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
                B, T = query_times.shape
                
                # Process all query times in parallel for both views
                motion_left_all = self._motion_head_multi(
                    1, [tok.float() for tok in dec_left], shape_left, query_times
                )
                motion_right_all = self._motion_head_multi(
                    2, [tok.float() for tok in dec_right], shape_right, query_times
                )
                
                # Organize motion predictions with dynamic keys
                motion_pred = {}
                
                for t_idx in range(T):
                    # Get the actual time value for this index
                    t_val = query_times[0, t_idx].item()  # Assuming all batches have same query times
                    
                    # Left view predicts motion to all times except 0
                    if abs(t_val - 0.0) > 1e-6:  # Not at t=0
                        key = f"l_to_{t_val:.3g}"  # Format nicely (e.g., 0.35 instead of 0.350)
                        motion_pred[key] = motion_left_all["map_pred"][:, t_idx]  # (B, H, W, 3)
                        if "map_pred_conf" in motion_left_all:
                            motion_pred[f"{key}_conf"] = motion_left_all["map_pred_conf"][:, t_idx]  # (B, H, W, 1)
                    
                    # Right view predicts motion to all times except 1
                    if abs(t_val - 1.0) > 1e-6:  # Not at t=1
                        key = f"r_to_{t_val:.3g}"
                        motion_pred[key] = motion_right_all["map_pred"][:, t_idx]  # (B, H, W, 3)
                        if "map_pred_conf" in motion_right_all:
                            motion_pred[f"{key}_conf"] = motion_right_all["map_pred_conf"][:, t_idx]  # (B, H, W, 1)

        # Rename right's 3D points to indicate they're in left's frame
        res_right["map_pred_in_left_frame"] = res_right.pop("map_pred")

        # Combine results into single dictionary
        combined_results = {}

        # Add left view 3D points
        for k, v in res_left.items():
            combined_results[f"left_{k}"] = v

        # Add right view 3D points
        for k, v in res_right.items():
            combined_results[f"right_{k}"] = v

        # Add motion predictions if computed
        if query_times is not None:
            combined_results["motion_pred"] = motion_pred

        # Add batch size for metrics calculation
        combined_results["batch_size"] = batch["left_image"].size(0)

        return combined_results
        # Output dictionary contains:
        # 'left_map_pred': (B, H, W, 3) - 3D points from left view
        # 'left_map_pred_conf': optional (B, H, W, 1) - confidence for left view points
        # 'right_map_pred_in_left_frame': (B, H, W, 3) - 3D points from right view in left frame
        # 'right_map_pred_conf': optional (B, H, W, 1) - confidence for right view points
        # 'motion_pred': dict with dynamic keys like "l_to_0.2", "r_to_0.35" etc - motion predictions
        # 'batch_size': integer - batch size for metrics calculation

    def get_loss(self, batch, outputs):
        """
        Compute total loss with aggressive memory optimization.
        Fixed to match unoptimized version behavior exactly.
        """
        device = batch["left_pm"].device
        alpha = 0.2
        
        # Extract query time once
        tq_mid = batch["query_times"][0, 0].item()
        
        # Process everything in chunks to avoid large intermediate tensors
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_details = {}
        
        # Define all loss computations
        loss_configs = [
            # Base point clouds
            {
                "name": "left",
                "gt": batch["left_pm"][..., :3],
                "pred": outputs["left_map_pred"],
                "valid": batch["left_pm"][..., 3] > 0,
                "conf": outputs.get("left_map_pred_conf", None),
                "is_base": True
            },
            {
                "name": "right", 
                "gt": batch["right_pm"][..., :3],
                "pred": outputs["right_map_pred_in_left_frame"],
                "valid": batch["right_pm"][..., 3] > 0,
                "conf": outputs.get("right_map_pred_conf", None),
                "is_base": True
            },
            # Motion point clouds
            {
                "name": "l2m",
                "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2m"][..., :3],
                "pred": outputs["left_map_pred"] + outputs["motion_pred"][f"l_to_{tq_mid:.3g}"][..., :3],
                "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2m"][..., 3] > 0),
                "conf": outputs["motion_pred"].get(f"l_to_{tq_mid:.3g}_conf", None),
                "is_base": False
            },
            {
                "name": "r2m",
                "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2m"][..., :3],
                "pred": outputs["right_map_pred_in_left_frame"] + outputs["motion_pred"][f"r_to_{tq_mid:.3g}"][..., :3],
                "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2m"][..., 3] > 0),
                "conf": outputs["motion_pred"].get(f"r_to_{tq_mid:.3g}_conf", None),
                "is_base": False
            },
            {
                "name": "l2r",
                "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2r"][..., :3],
                "pred": outputs["left_map_pred"] + outputs["motion_pred"]["l_to_1"][..., :3],
                "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2r"][..., 3] > 0),
                "conf": outputs["motion_pred"].get("l_to_1_conf", None),
                "is_base": False
            },
            {
                "name": "r2l",
                "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2l"][..., :3],
                "pred": outputs["right_map_pred_in_left_frame"] + outputs["motion_pred"]["r_to_0"][..., :3],
                "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2l"][..., 3] > 0),
                "conf": outputs["motion_pred"].get("r_to_0_conf", None),
                "is_base": False
            }
        ]
        
        # Compute normalization factors once for base PCs
        with torch.no_grad():
            # Extract base PCs and validity
            gt_left_pc = batch["left_pm"][..., :3]
            gt_right_pc = batch["right_pm"][..., :3]
            valid_left = batch["left_pm"][..., 3] > 0
            valid_right = batch["right_pm"][..., 3] > 0
            
            # GT normalization - using the FIXED version to match unoptimized
            gt_scale = self._compute_norm_factor_fixed(
                gt_left_pc, gt_right_pc, valid_left, valid_right
            )
        
        # Pred normalization (needs gradients)
        pred_scale = self._compute_norm_factor_fixed(
            outputs["left_map_pred"], 
            outputs["right_map_pred_in_left_frame"],
            valid_left, valid_right
        )
        
        # Process each loss component
        for cfg in loss_configs:
            # Compute single loss component
            loss_comp = self._compute_single_loss_fixed(
                cfg["gt"], cfg["pred"], cfg["valid"], cfg["conf"],
                gt_scale, pred_scale, alpha, device
            )
            
            # Accumulate loss
            total_loss = total_loss + loss_comp
            
            # Store details - FIXED to match unoptimized naming
            if cfg["conf"] is not None:
                loss_details[f"{cfg['name']}_conf"] = loss_comp.item()
                loss_details[f"{cfg['name']}_l2"] = 0.0  # L2 not used when conf available
            else:
                loss_details[f"{cfg['name']}_conf"] = 0.0
                loss_details[f"{cfg['name']}_l2"] = loss_comp.item()
            
            # Explicitly delete intermediate tensors if not base PC
            if not cfg["is_base"]:
                del cfg["gt"], cfg["pred"]
                if "valid" in cfg:
                    del cfg["valid"]
        
        loss_details["total_loss"] = total_loss.item()
        
        # Force garbage collection for large tensors
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return total_loss, loss_details


    def _compute_norm_factor_fixed(self, pc1, pc2, valid1, valid2):
        """
        Compute normalization factor matching unoptimized version exactly.
        Uses masked approach instead of extracting valid points.
        """
        batch_size = pc1.shape[0]
        device = pc1.device
        
        # Mask invalid points to 0 (matching unoptimized)
        pc1_masked = pc1.clone()
        pc2_masked = pc2.clone()
        pc1_masked[~valid1] = 0
        pc2_masked[~valid2] = 0
        
        # Stack and compute distances (matching unoptimized)
        all_pts = torch.cat([pc1_masked.flatten(1, 2), pc2_masked.flatten(1, 2)], dim=1)
        all_dis = all_pts.norm(dim=-1)
        
        # Count valid points
        nnz1 = valid1.sum(dim=(1, 2))
        nnz2 = valid2.sum(dim=(1, 2))
        
        # Average distance
        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
        norm_factor = norm_factor.clip(min=1e-8)
        
        # Expand to match PC dimensions
        while norm_factor.ndim < pc1.ndim:
            norm_factor = norm_factor.unsqueeze(-1)
        
        return norm_factor


    def _compute_single_loss_fixed(self, gt_pc, pred_pc, valid_mask, conf, 
                                gt_scale, pred_scale, alpha, device):
        """
        Compute loss for a single PC pair matching unoptimized behavior.
        Key fix: Only return confidence loss when available, otherwise return 0.
        """
        if not valid_mask.any():
            return torch.zeros(1, device=device, requires_grad=True)
        
        # Normalize full PCs first (like unoptimized)
        gt_pc_norm = gt_pc / gt_scale
        pred_pc_norm = pred_pc / pred_scale
        
        # Compute L2 distance for all pixels
        l2_dist = (pred_pc_norm - gt_pc_norm).norm(dim=-1)
        
        # Extract valid distances
        l2_dist_valid = l2_dist[valid_mask]
        
        # CRITICAL FIX: Match unoptimized behavior
        if conf is not None:
            # Extract valid confidence values
            conf_valid = conf[valid_mask]
            # Remove .squeeze(-1) unless you're sure about the shape
            # conf_valid = conf_valid.squeeze(-1)  
            
            # Compute confidence loss (no clamping to match unoptimized exactly)
            loss = (l2_dist_valid * conf_valid - alpha * torch.log(conf_valid)).mean()
        else:
            # Return 0 when no confidence (matching unoptimized bug/feature)
            loss = torch.zeros(1, device=device, requires_grad=True)
        
        return loss


    # ---------------------------------------------------------------------
    #  UPDATED: compute_metrics – static error + *point-cloud* motion error
    # ---------------------------------------------------------------------
    def compute_metrics(self, batch, outputs):
        """
        Returns:
            dict: {metric_name: float}
        """
        metrics = {}

        # ================================================================
        #  1. static 3-D point-cloud errors
        # ================================================================
        if "left_map_pred" in outputs and batch["left_pm"][..., 3].sum() > 0:
            mask = batch["left_pm"][..., 3] > 0
            metrics["left_3d_error"] = torch.norm(
                outputs["left_map_pred"][mask] - batch["left_pm"][..., :3][mask],
                dim=-1,
            ).mean().item()

        if "right_map_pred_in_left_frame" in outputs and batch["right_pm"][..., 3].sum() > 0:
            mask = batch["right_pm"][..., 3] > 0
            metrics["right_3d_error"] = torch.norm(
                outputs["right_map_pred_in_left_frame"][mask]
                - batch["right_pm"][..., :3][mask],
                dim=-1,
            ).mean().item()

        if "left_3d_error" in metrics and "right_3d_error" in metrics:
            metrics["avg_3d_error"] = (
                metrics["left_3d_error"] + metrics["right_3d_error"]
            ) / 2.0

        # ================================================================
        #  2. motion errors – compare *translated* point-clouds
        # ================================================================
        if "motion_pred" not in outputs or "motion_gt" not in batch:
            return metrics  # nothing to add

        tq_mid = (
            batch["query_times"][0, 0]  # (B,T)
            if batch["query_times"].dim() == 2
            else batch["query_times"][0]  # (T,)
        ).item()

        # Construct targets as base + GT motion (same as in loss computation)
        dir_cfg = {
            "l2m": {
                "pred_key": f"l_to_{tq_mid:.3g}",
                "src_pts": outputs["left_map_pred"],                          # (B,H,W,3)
                "tgt_pts": batch["left_pm"][..., :3] + batch["motion_gt"]["l2m"][..., :3],  # left base + l2m motion
                "base_valid": batch["left_pm"][..., 3] > 0,
                "motion_valid": batch["motion_gt"]["l2m"][..., 3] > 0,
            },
            "r2m": {
                "pred_key": f"r_to_{tq_mid:.3g}",
                "src_pts": outputs["right_map_pred_in_left_frame"],
                "tgt_pts": batch["right_pm"][..., :3] + batch["motion_gt"]["r2m"][..., :3],  # right base + r2m motion
                "base_valid": batch["right_pm"][..., 3] > 0,
                "motion_valid": batch["motion_gt"]["r2m"][..., 3] > 0,
            },
            "l2r": {
                "pred_key": "l_to_1",
                "src_pts": outputs["left_map_pred"],
                "tgt_pts": batch["left_pm"][..., :3] + batch["motion_gt"]["l2r"][..., :3],  # left base + l2r motion
                "base_valid": batch["left_pm"][..., 3] > 0,
                "motion_valid": batch["motion_gt"]["l2r"][..., 3] > 0,
            },
            "r2l": {
                "pred_key": "r_to_0",
                "src_pts": outputs["right_map_pred_in_left_frame"],
                "tgt_pts": batch["right_pm"][..., :3] + batch["motion_gt"]["r2l"][..., :3],  # right base + r2l motion
                "base_valid": batch["right_pm"][..., 3] > 0,
                "motion_valid": batch["motion_gt"]["r2l"][..., 3] > 0,
            },
        }

        motion_errs = []
        for name, cfg in dir_cfg.items():
            if cfg["pred_key"] not in outputs["motion_pred"]:
                continue  # prediction absent

            pred_disp = outputs["motion_pred"][cfg["pred_key"]]          # (B,H,W,3)
            pts_pred  = cfg["src_pts"] + pred_disp                       # translated cloud
            valid     = cfg["base_valid"] & cfg["motion_valid"]         # intersection validity

            if valid.sum() == 0:
                continue  # no valid GT for this direction

            err = torch.norm(
                pts_pred[valid] - cfg["tgt_pts"][valid], dim=-1
            ).mean().item()

            metrics[f"{name}_motion_pc_error"] = err
            motion_errs.append(err)

        if motion_errs:
            metrics["avg_motion_pc_error"] = sum(motion_errs) / len(motion_errs)

        return metrics



    def save_visualizations(self, batch, outputs, base_name, i=0, *args, **kwargs):
        from utils.viz import save_visualizations as save_viz
        save_viz(batch, outputs, base_name, i=i, *args, **kwargs)


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
                subprocess.run(["wget", pretrained_url, "-O", save_path], check=True)
            else:
                print(f"pretrained weights already exist at {save_path}.")

            # Load the checkpoint
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)

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

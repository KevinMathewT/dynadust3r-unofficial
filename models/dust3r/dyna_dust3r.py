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
import loaders.utils.geometry as geom
from loaders.utils.viz import visualize_image, visualize_pm, visualize_sequence_from_pms
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

    def compute_static_loss(self, criterion, batch, outputs, device):
        """
        Computes loss for static 3D reconstruction.

        Args:
            criterion: Loss criterion
            batch (dict): Batch data with ground truth
            outputs (dict): Model predictions
            device: PyTorch device

        Returns:
            tuple: (static_loss, static_loss_details)
        """
        batch_size = batch["left_image"].size(0)
        identity = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        gt_left = {
            "pts3d": batch["left_pm"][..., :3],
            "valid_mask": batch["left_pm"][..., 3] > 0,
            "camera_pose": identity,
        }

        gt_right = {
            "pts3d": batch["right_pm"][..., :3],
            "valid_mask": batch["right_pm"][..., 3] > 0,
            "camera_pose": identity,
        }

        pred_left = {"pts3d": outputs["left_map_pred"]}
        pred_right = {"pts3d_in_other_view": outputs["right_map_pred_in_left_frame"]}

        if "left_map_pred_conf" in outputs:
            pred_left["conf"] = outputs["left_map_pred_conf"]

        if "right_map_pred_conf" in outputs:
            pred_right["conf"] = outputs["right_map_pred_conf"]


        # ################# debug outputs #################
        # print(f"shape | gt_left['pts3d']: {gt_left['pts3d'].shape}, ")
        # print(f"shape | gt_right['pts3d']: {gt_right['pts3d'].shape}, ")
        # print(f"shape | pred_left['pts3d']: {pred_left['pts3d'].shape}, ")
        # print(f"shape | pred_right['pts3d_in_other_view']: {pred_right['pts3d_in_other_view'].shape}, ")
        # if "conf" in pred_left:
        #     print(f"shape | pred_left['conf']: {pred_left['conf'].shape}, ")
        # if "conf" in pred_right:
        #     print(f"shape | pred_right['conf']: {pred_right['conf'].shape}, ")
        # ##################################################

        return criterion(gt_left, gt_right, pred_left, pred_right)

    # ---------------------------------------------------------------------
    #  NEW: helper – pick the source 3-D points depending on teacher forcing
    # ---------------------------------------------------------------------
    def _get_src_pts(self, view: str, batch, outputs):
        """
        Args:
            view (str): 'left' or 'right'
        Returns:
            pts (Tensor)  # (B, H, W, 3)
        """
        if self.teacher_forcing:
            if view == "left":
                return batch["left_pm"][..., :3]         # (B,H,W,3)
            else:  # 'right'
                return batch["right_pm"][..., :3]        # (B,H,W,3)
        else:
            if view == "left":
                return outputs["left_map_pred"]          # (B,H,W,3)
            else:  # 'right'
                return outputs["right_map_pred_in_left_frame"]  # (B,H,W,3)

    # ---------------------------------------------------------------------
    #  UPDATED: compute_motion_loss – now handles l2m, r2m, l2r, r2l
    # ---------------------------------------------------------------------
    def compute_motion_loss(self, criterion, batch, outputs, device):
        """
        Sum motion losses for the four required directions:
          left→mid, right→mid, left→right, right→left
        """
        # sanity-checks ----------------------------------------------------
        if "motion_gt" not in batch:
            raise KeyError("batch must contain 'motion_gt' with keys l2m/r2m/l2r/r2l")
        if "motion_pred" not in outputs:
            raise KeyError("outputs must contain 'motion_pred' dict")

        motion_gt   = batch["motion_gt"]        # Dict[str, Tensor]
        motion_pred = outputs["motion_pred"]    # Dict[str, Tensor]

        # figure out the mid-frame tq so we can find the prediction keys ---
        tq_mid = (batch["query_times"][0, 0] if batch["query_times"].dim() == 2
                  else batch["query_times"][0]).item()
        # keys exactly as produced in model.forward
        pred_key = {
            "l2m": f"l_to_{tq_mid:.3g}",
            "r2m": f"r_to_{tq_mid:.3g}",
            "l2r": "l_to_1",
            "r2l": "r_to_0",
        }

        # make sure all predictions are present ---------------------------
        missing = [k for k, p in pred_key.items() if p not in motion_pred]
        if missing:
            raise KeyError(f"missing motion_pred keys for {missing}")

        # reusable identity pose ------------------------------------------
        B = batch["left_pm"].size(0)
        eye = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B,4,4)

        # helper to build gt / pred dicts for criterion -------------------
        def build_dict(pts3d, valid):
            return {
                "pts3d":     pts3d,            # (B,H,W,3)
                "valid_mask": valid,           # (B,H,W)
                "camera_pose": eye,            # (B,4,4)
            }

        total_motion_loss = torch.zeros((), device=device)     # scalar tensor
        loss_breakdown = {}

        # ------------------------------------------------------------------
        # Process motion losses: l2m+r2m paired (same timestep), l2r+r2l separate
        # ------------------------------------------------------------------
        
        # 1. Process l2m+r2m pair (both predict to same mid timestep - correct pairing)
        d1, v1 = "l2m", "left"
        d2, v2 = "r2m", "right"
        
        # Construct targets as base + GT motion (all motions are in left frame)
        base1 = batch["left_pm"][..., :3]   # l2m: left base
        base2 = batch["right_pm"][..., :3]  # r2m: right base  
        tgt1 = base1 + motion_gt[d1][..., :3]  # left + l2m motion
        tgt2 = base2 + motion_gt[d2][..., :3]  # right + r2m motion
        
        # Ground-truth dicts with intersection validity
        base_valid1 = (batch["left_pm"][..., 3] > 0)
        base_valid2 = (batch["right_pm"][..., 3] > 0)
        motion_valid1 = (motion_gt[d1][..., 3] > 0)
        motion_valid2 = (motion_gt[d2][..., 3] > 0)
        valid1 = base_valid1 & motion_valid1
        valid2 = base_valid2 & motion_valid2
        gt1 = build_dict(tgt1, valid1)
        gt2 = build_dict(tgt2, valid2)
        
        # Predictions
        src1 = self._get_src_pts(v1, batch, outputs)
        src2 = self._get_src_pts(v2, batch, outputs)
        delta1 = motion_pred[pred_key[d1]][..., :3]
        delta2 = motion_pred[pred_key[d2]][..., :3]
        pr1 = build_dict(src1 + delta1, valid1)
        pr2 = build_dict(src2 + delta2, valid2)
        
        # Optional confidence maps
        for d, pr in ((d1, pr1), (d2, pr2)):
            ck = f"{pred_key[d]}_conf"
            if ck in motion_pred:
                pr["conf"] = motion_pred[ck].squeeze(-1)
        
        # Compute paired loss for l2m+r2m
        loss_val, det = criterion(gt1, gt2, pr1, pr2)
        total_motion_loss += loss_val
        loss_breakdown[f"motion_pair_{d1}_{d2}"] = loss_val.detach()
        for k, v in det.items():
            loss_breakdown[f"{d1}_{d2}_{k}"] = v
        
        # Cleanup
        del gt1, gt2, pr1, pr2, src1, src2, delta1, delta2, valid1, valid2
        
        # 2. Process l2r individually (predicts to different timestep t=1)
        d1, v1 = "l2r", "left"
        
        # Construct target as base + GT motion
        base1 = batch["left_pm"][..., :3]   # l2r: left base
        tgt1 = base1 + motion_gt[d1][..., :3]  # left + l2r motion
        
        # Ground-truth dict with intersection validity (use dummy gt2 with same data for criterion compatibility)
        base_valid1 = (batch["left_pm"][..., 3] > 0)
        motion_valid1 = (motion_gt[d1][..., 3] > 0)
        valid1 = base_valid1 & motion_valid1
        gt1 = build_dict(tgt1, valid1)
        gt2 = build_dict(tgt1, valid1)  # Dummy - criterion expects two inputs
        
        # Predictions
        src1 = self._get_src_pts(v1, batch, outputs)
        delta1 = motion_pred[pred_key[d1]][..., :3]
        pr1 = build_dict(src1 + delta1, valid1)
        pr2 = build_dict(src1 + delta1, valid1)  # Dummy - same as pr1
        
        # Optional confidence map
        ck = f"{pred_key[d1]}_conf"
        if ck in motion_pred:
            pr1["conf"] = motion_pred[ck].squeeze(-1)
            pr2["conf"] = motion_pred[ck].squeeze(-1)  # Dummy
        
        # Compute individual loss for l2r (take only first component)
        loss_val, det = criterion(gt1, gt2, pr1, pr2)
        # Only use the first component since gt1==gt2 and pr1==pr2
        individual_loss = loss_val / 2.0  # Divide by 2 since criterion sums both components
        total_motion_loss += individual_loss
        loss_breakdown[f"motion_individual_{d1}"] = individual_loss.detach()
        # Extract only first component from details
        for k, v in det.items():
            if k.endswith('_1'):  # Only take the first component
                loss_breakdown[f"{d1}_{k}"] = v
        
        # Cleanup
        del gt1, gt2, pr1, pr2, src1, delta1, valid1
        
        # 3. Process r2l individually (predicts to different timestep t=0)
        d1, v1 = "r2l", "right"
        
        # Construct target as base + GT motion
        base1 = batch["right_pm"][..., :3]  # r2l: right base
        tgt1 = base1 + motion_gt[d1][..., :3]  # right + r2l motion
        
        # Ground-truth dict with intersection validity (use dummy gt2 with same data for criterion compatibility)
        base_valid1 = (batch["right_pm"][..., 3] > 0)
        motion_valid1 = (motion_gt[d1][..., 3] > 0)
        valid1 = base_valid1 & motion_valid1
        gt1 = build_dict(tgt1, valid1)
        gt2 = build_dict(tgt1, valid1)  # Dummy - criterion expects two inputs
        
        # Predictions
        src1 = self._get_src_pts(v1, batch, outputs)
        delta1 = motion_pred[pred_key[d1]][..., :3]
        pr1 = build_dict(src1 + delta1, valid1)
        pr2 = build_dict(src1 + delta1, valid1)  # Dummy - same as pr1
        
        # Optional confidence map
        ck = f"{pred_key[d1]}_conf"
        if ck in motion_pred:
            pr1["conf"] = motion_pred[ck].squeeze(-1)
            pr2["conf"] = motion_pred[ck].squeeze(-1)  # Dummy
        
        # Compute individual loss for r2l (take only first component)
        loss_val, det = criterion(gt1, gt2, pr1, pr2)
        # Only use the first component since gt1==gt2 and pr1==pr2
        individual_loss = loss_val / 2.0  # Divide by 2 since criterion sums both components
        total_motion_loss += individual_loss
        loss_breakdown[f"motion_individual_{d1}"] = individual_loss.detach()
        # Extract only first component from details
        for k, v in det.items():
            if k.endswith('_1'):  # Only take the first component
                loss_breakdown[f"{d1}_{k}"] = v
        
        # Cleanup
        del gt1, gt2, pr1, pr2, src1, delta1, valid1
        # ------------------------------------------------------------------

        return total_motion_loss, loss_breakdown

    # ---------------------------------------------------------------------
    #  UPDATED: get_loss – sums static + *all four* motion losses
    # ---------------------------------------------------------------------
    def get_loss(self, criterion, batch, outputs):
        device = batch["left_pm"].device

        static_loss, static_det = self.compute_static_loss(criterion,
                                                           batch, outputs, device)

        motion_loss, motion_det = self.compute_motion_loss(criterion,
                                                           batch, outputs, device)

        total_loss = static_loss + motion_loss  # 1 : 1 weighting; tweak if needed

        # flatten details --------------------------------------------------
        details = {**static_det, **motion_det,
                   "static_loss": static_loss.item(),
                   "motion_loss": motion_loss.item()}

        return total_loss, details


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
        # -----------------------------------------------------------
        # small helpers
        # -----------------------------------------------------------
        def first_to_numpy(x):
            """
            Accepts:  numpy array | torch tensor | list
            Returns:  numpy array without batch dim
            """
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(x, (list, tuple)):
                x = np.asarray(x)
            return x[0] if x.ndim == 3 else x  # strip batch if present

        def img_to_uint8(img_t):                  # (C,H,W) → (H,W,3)
            img = img_t.permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5 + 0.5) * 255.0
            return np.clip(img, 0, 255).astype(np.uint8)

        def depth_to_heatmap(z, z_min, z_max, cmap=cm.prism):                  # (H,W) → (H,W,3) uint8
            valid = z > 0
            if not np.any(valid):
                return np.zeros((*z.shape, 3), np.uint8)
            zv     = np.where(valid, z, np.nan)
            norm   = (zv - z_min) / (z_max - z_min + 1e-6)   # shared min/max
            hm_rgb = cmap(norm)[:, :, :3]                # prism colormap
            hm_rgb[~valid] = 0
            return (hm_rgb * 255).astype(np.uint8)
        
        def disparity_to_heatmap(d, d_min, d_max, cmap=cm.turbo):
            valid = d > 0
            if not np.any(valid):
                return np.zeros((*d.shape, 3), np.uint8)
            dv     = np.where(valid, d, np.nan)
            norm   = (dv - d_min) / (d_max - d_min + 1e-6)   # shared min/max
            hm_rgb = cmap(norm)[:, :, :3]                    # turbo colormap
            hm_rgb[~valid] = 0
            return (hm_rgb * 255).astype(np.uint8)


        def conf_to_grayscale(conf):              # (H,W) → (H,W,3) uint8
            """Convert confidence map (1 to inf) to grayscale image.
            1 → black (0), inf → white (255), log scale for values in between."""
            # Handle invalid/missing confidence
            if conf is None or conf.size == 0:
                return None
            
            # Clip to valid range [1, inf) and apply log transform
            conf_clipped = np.maximum(conf, 1.0)
            log_conf = np.log(conf_clipped)       # log(1) = 0, log(inf) = inf
            
            # Find valid range for normalization
            valid_mask = np.isfinite(log_conf)
            if not np.any(valid_mask):
                return np.zeros((*conf.shape, 3), np.uint8)
            
            # Normalize to [0, 1] range
            min_val = np.min(log_conf[valid_mask])
            max_val = np.max(log_conf[valid_mask])
            
            if max_val - min_val < 1e-6:          # all same value
                gray_val = 0 if min_val == 0 else 127
                gray = np.full(conf.shape, gray_val, dtype=np.uint8)
            else:
                norm = (log_conf - min_val) / (max_val - min_val)
                norm = np.where(valid_mask, norm, 0)
                gray = (norm * 255).astype(np.uint8)
            
            # Convert to 3-channel for consistency with other visualizations
            return np.stack([gray, gray, gray], axis=-1)

        def conf_to_heatmap(conf, cmap=cm.plasma):
            """Convert confidence map (1 to inf) to a colored heatmap.
            1 → darkest color, inf → brightest color, log scale for values in between."""
            if conf is None or conf.size == 0:
                return None
            
            conf_clipped = np.maximum(conf, 1.0)
            log_conf = np.log(conf_clipped)
            
            valid_mask = np.isfinite(log_conf)
            if not np.any(valid_mask):
                return np.zeros((*conf.shape, 3), np.uint8)
                
            min_val = np.min(log_conf[valid_mask])
            max_val = np.max(log_conf[valid_mask])
            
            if max_val - min_val < 1e-6:
                norm = np.full(conf.shape, 0.0 if min_val == 0 else 0.5)
            else:
                norm = (log_conf - min_val) / (max_val - min_val)
                norm = np.where(valid_mask, norm, 0)

            heatmap_rgb = cmap(norm)[:, :, :3] # drop alpha
            heatmap_rgb[~valid_mask] = 0
            return (heatmap_rgb * 255).astype(np.uint8)

        def motion_magnitude_to_grayscale(motion_vec, validity=None): # (H,W,3) → (H,W,3) uint8
            """Convert motion vector magnitude to grayscale image.
            0 → black (0), max → white (255)."""
            # Compute L2 norm of motion vectors
            magnitude = np.linalg.norm(motion_vec, axis=-1) # (H,W)
            
            # Apply validity mask if provided
            if validity is not None:
                magnitude = magnitude * validity
            
            # Normalize to [0, 1] range
            valid = magnitude > 0
            if not np.any(valid):
                return np.zeros((*magnitude.shape, 3), np.uint8)
            
            max_val = np.max(magnitude[valid])
            if max_val < 1e-6: # essentially no motion
                gray = np.zeros(magnitude.shape, dtype=np.uint8)
            else:
                norm = magnitude / max_val
                gray = (norm * 255).astype(np.uint8)
            
            # Convert to 3-channel
            return np.stack([gray, gray, gray], axis=-1)

        def visualize_3d_motion_field(points, motion_vectors, rgb_image, validity=None, 
                                    subsample_factor=50, arrow_scale=1.0, 
                                    view_angles=(30, -60), figsize=(12, 10)):
            """Visualize 3D motion field using arrows (scene flow visualization).
            
            Args:
                points: (H,W,3) array of 3D points
                motion_vectors: (H,W,3) array of 3D motion vectors
                rgb_image: (H,W,3) RGB image for coloring points
                validity: (H,W) optional validity mask
                subsample_factor: downsample points for visualization (default 50)
                arrow_scale: scale factor for arrow length (default 1.0)
                view_angles: (elevation, azimuth) viewing angles in degrees
                figsize: figure size tuple
                
            Returns:
                (H_fig, W_fig, 3) uint8 RGB image of the visualization
            """
            H, W = points.shape[:2]
            
            # Get valid points mask
            valid_z = points[:, :, 2] > 0
            if validity is not None:
                valid_mask = valid_z & (validity > 0)
            else:
                valid_mask = valid_z
                
            # Flatten and subsample
            ys, xs = np.where(valid_mask)
            
            # Subsample uniformly
            n_points = len(xs)
            if n_points > subsample_factor:
                # Random but reproducible subsampling
                np.random.seed(42)
                indices = np.random.choice(n_points, n_points // subsample_factor, replace=False)
                xs = xs[indices]
                ys = ys[indices]
            
            # Extract subsampled data
            pts = points[ys, xs]  # (N, 3)
            vecs = motion_vectors[ys, xs]  # (N, 3)
            colors = rgb_image[ys, xs] / 255.0  # (N, 3) normalized to [0,1]
            
            # Compute motion magnitudes for coloring arrows
            magnitudes = np.linalg.norm(vecs, axis=1)
            
            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points as scatter
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                    c=colors, s=20, alpha=0.6, edgecolors='none')
            
            # Normalize arrow colors by magnitude
            if magnitudes.max() > 0:
                norm = Normalize(vmin=0, vmax=np.percentile(magnitudes[magnitudes > 0], 95))
                cmap = cm.plasma
                arrow_colors = cmap(norm(magnitudes))
            else:
                arrow_colors = np.zeros((len(magnitudes), 4))
                arrow_colors[:, 3] = 1.0  # Set alpha
            
            # Plot motion vectors as arrows
            # Only plot arrows with significant motion
            motion_threshold = 0.01 * np.max(magnitudes) if np.max(magnitudes) > 0 else 0
            significant_motion = magnitudes > motion_threshold
            
            if np.any(significant_motion):
                sig_pts = pts[significant_motion]
                sig_vecs = vecs[significant_motion] * arrow_scale
                sig_colors = arrow_colors[significant_motion]
                
                ax.quiver(sig_pts[:, 0], sig_pts[:, 1], sig_pts[:, 2],
                        sig_vecs[:, 0], sig_vecs[:, 1], sig_vecs[:, 2],
                        color=sig_colors, arrow_length_ratio=0.2, 
                        linewidth=2, alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Scene Flow Visualization\n(Points colored by RGB, Arrows colored by motion magnitude)')
            
            # Set viewing angle
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
            
            # Equal aspect ratio
            max_range = np.array([
                pts[:, 0].max() - pts[:, 0].min(),
                pts[:, 1].max() - pts[:, 1].min(),
                pts[:, 2].max() - pts[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
            mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
            mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add colorbar for motion magnitude
            if np.any(significant_motion):
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(magnitudes)
                cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
                cbar.set_label('Motion Magnitude', rotation=270, labelpad=15)
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            # Read image from buffer
            img = Image.open(buf)
            img_array = np.array(img)[:, :, :3]  # Remove alpha channel if present
            
            return img_array
        
        def create_motion_summary_figure(left_motion_gt, left_motion_pred, 
                                        right_motion_gt, right_motion_pred,
                                        left_pm_gt, right_pm_gt,
                                        left_rgb, right_rgb,
                                        left_validity=None, right_validity=None):
            """Create a summary figure with 2x2 grid of 3D motion visualizations."""
            # Create individual 3D visualizations
            viz_left_gt = visualize_3d_motion_field(
                left_pm_gt, left_motion_gt, left_rgb, left_validity,
                subsample_factor=30, view_angles=(20, -45)
            )
            
            viz_left_pred = visualize_3d_motion_field(
                left_pm_gt, left_motion_pred, left_rgb, None,
                subsample_factor=30, view_angles=(20, -45)
            )
            
            viz_right_gt = visualize_3d_motion_field(
                right_pm_gt, right_motion_gt, right_rgb, right_validity,
                subsample_factor=30, view_angles=(20, -135)
            )
            
            viz_right_pred = visualize_3d_motion_field(
                right_pm_gt, right_motion_pred, right_rgb, None,
                subsample_factor=30, view_angles=(20, -135)
            )
            
            # Create summary figure
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.1)
            
            # Plot each visualization
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(viz_left_gt)
            ax1.set_title('Left Camera - Ground Truth Motion', fontsize=14, pad=10)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(viz_left_pred)
            ax2.set_title('Left Camera - Predicted Motion', fontsize=14, pad=10)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(viz_right_gt)
            ax3.set_title('Right Camera - Ground Truth Motion', fontsize=14, pad=10)
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.imshow(viz_right_pred)
            ax4.set_title('Right Camera - Predicted Motion', fontsize=14, pad=10)
            ax4.axis('off')
            
            # Add main title
            fig.suptitle('3D Scene Flow Visualization\n(Arrows show 3D motion vectors, colored by magnitude)', 
                        fontsize=16, y=0.98)
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            # Read image from buffer
            img = Image.open(buf)
            img_array = np.array(img)[:, :, :3]  # Remove alpha channel if present
            
            return img_array

        def pointmap_to_pointcloud(pm, max_points=300000): # (H,W,3 or 4) → (N,3)
            """Convert point map to point cloud for wandb Object3D.
            Filters out invalid points (z <= 0) and downsamples if needed."""
            # Extract xyz coordinates
            xyz = pm[:, :, :3] if pm.shape[-1] >= 3 else pm
            
            # Get valid mask (points with positive z)
            valid_mask = xyz[:, :, 2] > 0
            
            # Flatten and filter valid points
            points = xyz[valid_mask]  # (N, 3)
            
            # Downsample if exceeding max points (wandb limit)
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
            
            return points

        def pointmap_to_colored_pointcloud(pm, rgb_image, K=None, max_points=300_000):
            """
            Convert point map to coloured point cloud.
            Returns (N,6) array:  [x, y, z, r, g, b].
            """
            H, W = pm.shape[:2]

            # xyz & valid mask
            xyz = pm[..., :3]
            valid_mask = xyz[..., 2] > 0
            ys, xs = np.where(valid_mask)
            if len(xs) == 0:
                return np.empty((0, 6), np.float32)

            points_3d = xyz[ys, xs]                     # (N,3)

            # if intrinsics supplied, keep a sanity-check projection (optional)
            if K is not None:
                pts_h = points_3d / (points_3d[:, 2:3] + 1e-8)
                proj   = (K @ pts_h.T).T
                us = np.round(proj[:, 0]).astype(int)
                vs = np.round(proj[:, 1]).astype(int)
            else:
                us, vs = xs, ys                         # already pixel-aligned

            in_bounds = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
            points_3d = points_3d[in_bounds]
            us, vs    = us[in_bounds], vs[in_bounds]

            colours = rgb_image[vs, us]                 # (N,3) uint8
            pc = np.concatenate([points_3d, colours], axis=1)  # (N,6)

            # optional down-sampling
            if len(pc) > max_points:
                idx = np.random.choice(len(pc), max_points, replace=False)
                pc = pc[idx]

            return pc


        def transform_pointmap_to_right_frame(pm_left, E_L, E_R):
            """Transform point map from left camera frame to right camera frame."""
            H, W, _ = pm_left.shape
            # Reshape to (HW, 3) for transformation
            pts_left = pm_left.reshape(-1, 3)
            # Add homogeneous coordinate
            pts_left_hom = np.concatenate([pts_left, np.ones((pts_left.shape[0], 1))], axis=1)
            
            # Transform: left camera → world → right camera
            pts_world = pts_left_hom @ np.linalg.inv(E_L).T
            pts_right_hom = pts_world @ E_R.T
            
            # Extract xyz and reshape back
            pts_right = pts_right_hom[:, :3].reshape(H, W, 3)
            
            # Keep invalid points as is
            valid_mask = pm_left[:, :, 2] > 0
            pts_right[~valid_mask] = 0
            
            return pts_right

        def make_flip_gif(img_a: np.ndarray, img_b: np.ndarray, fps: int = 1):
            """
            Create a 2-frame gif that toggles A ↔ B every `1/fps` seconds.

            img_* : uint8 (H, W, 3) RGB
            returns: wandb.Video ready to log
            """
            assert img_a.shape == img_b.shape and img_a.ndim == 3
            # (T, C, H, W)  uint8
            frames = np.stack([img_a.transpose(2,0,1),   # frame-0
                            img_b.transpose(2,0,1)],  # frame-1
                            axis=0)                    # (2, 3, H, W)
            return wandb.Video(frames, fps=fps, format="gif")

        # -----------------------------------------------------------
        # tiny helpers for 2-D vector overlay
        # -----------------------------------------------------------
        def project_cam_pts(K, xyz):                    # (N,3) → (N,2)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x, y, z = xyz.T
            z_pos = z > 1e-6
            uv = np.zeros((len(z), 2), dtype=np.int32)
            uv[:] = -1
            if np.any(z_pos):
                u = fx * x[z_pos] / z[z_pos] + cx
                v = fy * y[z_pos] / z[z_pos] + cy
                uv[z_pos, 0] = np.round(u).astype(int)
                uv[z_pos, 1] = np.round(v).astype(int)
            return uv                                         # (N,2)

        def draw_gradient_line(img, p0, p1, cmap=cm.viridis, n_seg=8, thick=1):
            """Draw solid fluorescent green line p0→p1; p0/p1 are (u,v) int pairs.
            FIXED: Now properly handles RGB images with OpenCV."""
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Fluorescent green in BGR format
            fluorescent_green_bgr = (0, 255, 0)  # BGR for cv2
            
            # Draw single line instead of segments
            cv2.line(img_bgr,
                    tuple(p0), tuple(p1),
                    color=fluorescent_green_bgr, thickness=thick, lineType=cv2.LINE_AA)
            
            # Convert back to RGB
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB, dst=img)

        # -----------------------------------------------------------
        # grab camera for *first* item in batch
        # -----------------------------------------------------------
        K_L, E_L = batch["cam"]
        K_M, E_M = batch["cam_mid"]
        K_R, E_R = batch["cam_right"]

        K_L = first_to_numpy(K_L)     # (3,3)
        E_L = first_to_numpy(E_L)     # (4,4)
        K_M = first_to_numpy(K_M)
        E_M = first_to_numpy(E_M)
        K_R = first_to_numpy(K_R)
        E_R = first_to_numpy(E_R)

        # -----------------------------------------------------------
        # get the three RGB inputs (selected sample)
        # -----------------------------------------------------------
        left_rgb  = img_to_uint8(batch["left_image"][i])
        mid_rgb   = img_to_uint8(batch["mid_image"][i])
        right_rgb = img_to_uint8(batch["right_image"][i])

        # -----------------------------------------------------------
        # left-view depths
        # -----------------------------------------------------------
        z_left_gt   = batch["left_pm"][i, :, :, 2].cpu().numpy()              # (H,W)
        z_left_pred = outputs["left_map_pred"][i, :, :, 2].detach().cpu().numpy()

        # -----------------------------------------------------------
        # right-view GT & pred are in *left* coords – convert to right
        # -----------------------------------------------------------
        z_r_gt_left   = batch["right_pm"][i, :, :, 2].cpu().numpy()
        z_r_pred_left = outputs["right_map_pred_in_left_frame"][i, :, :, 2].detach().cpu().numpy()

        # -----------------------------------------------------------
        # confidence maps (predictions only)
        # -----------------------------------------------------------
        # Extract confidence values if they exist
        conf_left_pred = None
        conf_right_pred = None
        
        if "left_map_pred_conf" in outputs:
            conf_left_pred = outputs["left_map_pred_conf"][i, :, :].detach().cpu().numpy() # (H,W)
        
        if "right_map_pred_conf" in outputs:
            # Note: right confidence is already in right camera frame, no transformation needed
            conf_right_pred = outputs["right_map_pred_conf"][i, :, :].detach().cpu().numpy() # (H,W)
        
        # Convert to grayscale images
        gray_conf_left = conf_to_grayscale(conf_left_pred) if conf_left_pred is not None else None
        gray_conf_right = conf_to_grayscale(conf_right_pred) if conf_right_pred is not None else None

        # -----------------------------------------------------------
        # 3D point clouds
        # -----------------------------------------------------------
        # Extract full point maps (GT and pred)
        left_pm_gt = batch["left_pm"][i, :, :, :3].cpu().numpy() # (H,W,3)
        left_pm_pred = outputs["left_map_pred"][i, :, :, :].detach().cpu().numpy() # (H,W,3)

        # Right point maps are in left camera frame - need transformation
        right_pm_gt_left = batch["right_pm"][i, :, :, :3].cpu().numpy() # (H,W,3)
        right_pm_pred_left = outputs["right_map_pred_in_left_frame"][i, :, :, :].detach().cpu().numpy() # (H,W,3)

        # Compute scale factor to normalize predictions
        H, W = left_pm_gt.shape[:2]
        gt1 = torch.from_numpy(left_pm_gt[None, ...]).float()
        gt2 = torch.from_numpy(right_pm_gt_left[None, ...]).float()
        pr1 = torch.from_numpy(left_pm_pred[None, ...]).float()
        pr2 = torch.from_numpy(right_pm_pred_left[None, ...]).float()

        left_valid = batch["left_pm"][i, :, :, 3].cpu().numpy() > 0
        right_valid = batch["right_pm"][i, :, :, 3].cpu().numpy() > 0
        valid1 = torch.from_numpy(left_valid)[None, ..., None].expand(-1, H, W, 3).contiguous()
        valid2 = torch.from_numpy(right_valid)[None, ..., None].expand(-1, H, W, 3).contiguous()

        _, _, pred_factor = normalize_pointcloud(
            pr1, pr2, norm_mode='avg_dis', valid1=valid1, valid2=valid2, ret_factor=True
        )
        _, _, gt_factor = normalize_pointcloud(
            gt1, gt2, norm_mode='avg_dis', valid1=valid1, valid2=valid2, ret_factor=True
        )

        scale = (gt_factor / pred_factor).item()

        # Apply scale before transformation
        left_pm_pred = left_pm_pred * scale
        right_pm_pred_left = right_pm_pred_left * scale

        # Transform right point maps to right camera frame
        right_pm_gt = transform_pointmap_to_right_frame(right_pm_gt_left, E_L, E_R)
        right_pm_pred = transform_pointmap_to_right_frame(right_pm_pred_left, E_L, E_R)


        def left_z_to_right_z(z_left):
            """Convert a per-pixel z map in left-cam coords → right-cam depths."""
            H, W            = z_left.shape
            ys, xs          = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            fx, fy          = K_L[0, 0], K_L[1, 1]
            cx, cy          = K_L[0, 2], K_L[1, 2]

            # 3-D pts in left camera frame
            X = (xs - cx) * z_left / fx
            Y = (ys - cy) * z_left / fy
            Z = z_left
            ptsL = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1).reshape(-1, 4) # (HW,4)

            # world → right-cam
            world = ptsL @ np.linalg.inv(E_L).T            # (HW,4)
            ptsR  = world @ E_R.T                           # (HW,4)
            Zr    = ptsR[:, 2].reshape(H, W)

            # keep invalid pixels at 0
            Zr[z_left <= 0] = 0
            return Zr

        z_right_gt   = left_z_to_right_z(z_r_gt_left)
        z_right_pred = left_z_to_right_z(z_r_pred_left)

        # -----------------------------------------------------------
        # depth & disparity heat-maps  (robust min/max, ignore z==0)
        # -----------------------------------------------------------
        P_ZMAX = 99                    # percentile for clipping far depths

        # scale predicted depths so GT & pred share the same metric scale
        z_left_pred  *= scale
        z_right_pred *= scale

        # ---------- collect *valid* depths (>0) ----------
        valid_depths = [
            z_left_gt [z_left_gt  > 0],
            z_left_pred[z_left_pred > 0],
            z_right_gt[z_right_gt > 0],
            z_right_pred[z_right_pred > 0],
        ]
        z_stack = np.concatenate(valid_depths)
        z_min   = float(z_stack.min())
        z_max   = float(np.percentile(z_stack, P_ZMAX))

        # ---------- prism-coloured depth maps ----------
        hm_left_gt    = depth_to_heatmap(z_left_gt,  z_min, z_max)
        hm_left_pred  = depth_to_heatmap(z_left_pred, z_min, z_max)
        hm_right_gt   = depth_to_heatmap(z_right_gt, z_min, z_max)
        hm_right_pred = depth_to_heatmap(z_right_pred, z_min, z_max)

        # -----------------------------------------------------------
        # disparity (turbo) – uses the scaled depths above
        # -----------------------------------------------------------
        fx        = K_L[0, 0]
        baseline  = np.linalg.norm(E_L[:3, 3] - E_R[:3, 3])      # metres
        def z_to_disp(z):
            disp = np.zeros_like(z, dtype=np.float32)
            mask = z > 0
            disp[mask] = fx * baseline / z[mask]
            return disp, mask

        disp_left_gt,   mask_l_gt  = z_to_disp(z_left_gt)
        disp_left_pred, mask_l_pr  = z_to_disp(z_left_pred)
        disp_right_gt,  mask_r_gt  = z_to_disp(z_right_gt)
        disp_right_pred,mask_r_pr  = z_to_disp(z_right_pred)

        # robust min/max over valid disparities only
        valid_disps = np.concatenate([
            disp_left_gt [mask_l_gt],
            disp_left_pred[mask_l_pr],
            disp_right_gt[mask_r_gt],
            disp_right_pred[mask_r_pr],
        ])
        d_min = float(valid_disps.min())
        d_max = float(valid_disps.max())

        # ---------- turbo-coloured disparity maps ----------
        hm_disp_left_gt    = disparity_to_heatmap(disp_left_gt,  d_min, d_max)
        hm_disp_left_pred  = disparity_to_heatmap(disp_left_pred, d_min, d_max)
        hm_disp_right_gt   = disparity_to_heatmap(disp_right_gt, d_min, d_max)
        hm_disp_right_pred = disparity_to_heatmap(disp_right_pred, d_min, d_max)


        # -----------------------------------------------------------
        # motion maps (GT & pred)  — handles l2m, r2m, l2r, r2l
        # -----------------------------------------------------------
        motion_gt   = batch["motion_gt"]           # dict[str → (H,W,4)]
        motion_pred = outputs["motion_pred"]       # dict[str → (B,H,W,3)]

        # ---- keys produced by forward() ---------------------------------
        tq_mid = ( batch["query_times"][0, 0]
                    if batch["query_times"].dim() == 2
                    else batch["query_times"][0] ).item()
        k_l2m = f"l_to_{tq_mid:.3g}"           # left → mid
        k_r2m = f"r_to_{tq_mid:.3g}"           # right→ mid
        k_l2r = "l_to_1"                       # left → right
        k_r2l = "r_to_0"                       # right→ left

        # ---- helper to split xyz / validity ------------------------------
        def split_xyz_val(arr4):               # (H,W,4) → xyz(…3), val(…)
            return arr4[..., :3], arr4[..., 3]

        # ---- ground-truth motions ----------------------------------------
        left_motion_gt,  left_motion_val  = split_xyz_val(motion_gt["l2m"][i].cpu().numpy())
        right_motion_gt, right_motion_val = split_xyz_val(motion_gt["r2m"][i].cpu().numpy())
        l2r_motion_gt,   l2r_motion_val   = split_xyz_val(motion_gt["l2r"][i].cpu().numpy())
        r2l_motion_gt,   r2l_motion_val   = split_xyz_val(motion_gt["r2l"][i].cpu().numpy())

        # ---- predicted displacements -------------------------------------
        left_motion_pred  = motion_pred[k_l2m][i].detach().cpu().numpy()     # (H,W,3)
        right_motion_pred = motion_pred[k_r2m][i].detach().cpu().numpy()
        l2r_motion_pred   = motion_pred[k_l2r][i].detach().cpu().numpy()
        r2l_motion_pred   = motion_pred[k_r2l][i].detach().cpu().numpy()

        # ---- scale predictions to match depth normalisation --------------
        left_motion_pred  *= scale
        right_motion_pred *= scale
        l2r_motion_pred   *= scale
        r2l_motion_pred   *= scale

        # ---- grayscale magnitude images ----------------------------------
        gray_motion_left_gt   = motion_magnitude_to_grayscale(left_motion_gt,  left_motion_val)
        gray_motion_left_pred = motion_magnitude_to_grayscale(left_motion_pred)
        gray_motion_right_gt  = motion_magnitude_to_grayscale(right_motion_gt, right_motion_val)
        gray_motion_right_pred= motion_magnitude_to_grayscale(right_motion_pred)

        gray_motion_l2r_gt    = motion_magnitude_to_grayscale(l2r_motion_gt,  l2r_motion_val)
        gray_motion_l2r_pred  = motion_magnitude_to_grayscale(l2r_motion_pred)
        gray_motion_r2l_gt    = motion_magnitude_to_grayscale(r2l_motion_gt,  r2l_motion_val)
        gray_motion_r2l_pred  = motion_magnitude_to_grayscale(r2l_motion_pred)

        # -----------------------------------------------------------
        # 3-D scene-flow visualisations  (l2m, r2m, l2r, r2l)
        # -----------------------------------------------------------
        viz_left_gt = visualize_3d_motion_field(
            left_pm_gt, left_motion_gt, left_rgb, left_motion_val,
            subsample_factor=30, view_angles=(20, -45)
        )
        viz_left_pred = visualize_3d_motion_field(
            left_pm_gt, left_motion_pred, left_rgb, None,
            subsample_factor=30, view_angles=(20, -45)
        )

        viz_right_gt = visualize_3d_motion_field(
            right_pm_gt, right_motion_gt, right_rgb, right_motion_val,
            subsample_factor=30, view_angles=(20, -135)
        )
        viz_right_pred = visualize_3d_motion_field(
            right_pm_gt, right_motion_pred, right_rgb, None,
            subsample_factor=30, view_angles=(20, -135)
        )

        viz_l2r_gt = visualize_3d_motion_field(
            left_pm_gt, l2r_motion_gt, left_rgb, l2r_motion_val,
            subsample_factor=30, view_angles=(20, -45)
        )
        viz_l2r_pred = visualize_3d_motion_field(
            left_pm_gt, l2r_motion_pred, left_rgb, None,
            subsample_factor=30, view_angles=(20, -45)
        )
        viz_r2l_gt = visualize_3d_motion_field(
            right_pm_gt, r2l_motion_gt, right_rgb, r2l_motion_val,
            subsample_factor=30, view_angles=(20, -135)
        )
        viz_r2l_pred = visualize_3d_motion_field(
            right_pm_gt, r2l_motion_pred, right_rgb, None,
            subsample_factor=30, view_angles=(20, -135)
        )

        # -----------------------------------------------------------
        # motion confidence maps (predictions only)
        # -----------------------------------------------------------
        conf_motion_left_pred = motion_pred.get(f"{k_l2m}_conf", None)
        conf_motion_right_pred= motion_pred.get(f"{k_r2m}_conf", None)
        conf_motion_l2r_pred  = motion_pred.get(f"{k_l2r}_conf", None)
        conf_motion_r2l_pred  = motion_pred.get(f"{k_r2l}_conf", None)
        
        # motion confidence maps (predictions only)  ← keep this comment
        conf_motion_left_pred  = None if conf_motion_left_pred  is None else conf_motion_left_pred[i].detach().cpu().numpy()
        conf_motion_right_pred = None if conf_motion_right_pred is None else conf_motion_right_pred[i].detach().cpu().numpy()
        conf_motion_l2r_pred   = None if conf_motion_l2r_pred   is None else conf_motion_l2r_pred[i].detach().cpu().numpy()
        conf_motion_r2l_pred   = None if conf_motion_r2l_pred   is None else conf_motion_r2l_pred[i].detach().cpu().numpy()


        hm_motion_conf_left  = conf_to_grayscale(conf_motion_left_pred)   if conf_motion_left_pred  is not None else None
        hm_motion_conf_right = conf_to_grayscale(conf_motion_right_pred)  if conf_motion_right_pred is not None else None
        hm_motion_conf_l2r   = conf_to_grayscale(conf_motion_l2r_pred)    if conf_motion_l2r_pred   is not None else None
        hm_motion_conf_r2l   = conf_to_grayscale(conf_motion_r2l_pred)    if conf_motion_r2l_pred   is not None else None
        # -----------------------------------------------------------

        # Convert to colored point clouds for wandb
        # Left point clouds use left image for colors
        pc_left_gt = pointmap_to_colored_pointcloud(left_pm_gt, left_rgb, K_L)
        pc_left_pred = pointmap_to_colored_pointcloud(left_pm_pred, left_rgb, K_L)
        
        # Right point clouds (now in right camera frame) use right image for colors
        pc_right_gt = pointmap_to_colored_pointcloud(right_pm_gt, right_rgb, K_R)
        pc_right_pred = pointmap_to_colored_pointcloud(right_pm_pred, right_rgb, K_R)

        flip_left_gt  = make_flip_gif(left_rgb, hm_left_gt)   # input ↔ gt-depth
        flip_left  = make_flip_gif(left_rgb,  hm_left_pred)   # input ↔ pred-depth
        flip_right_gt = make_flip_gif(right_rgb, hm_right_gt)  # input ↔ gt-depth
        flip_right = make_flip_gif(right_rgb, hm_right_pred)  # input ↔ pred-depth
        
        # New flip gifs
        flip_left_depth_gt_pred = make_flip_gif(hm_left_gt, hm_left_pred)    # gt-depth ↔ pred-depth
        flip_right_depth_gt_pred = make_flip_gif(hm_right_gt, hm_right_pred)  # gt-depth ↔ pred-depth
        flip_left_right_rgb = make_flip_gif(left_rgb, right_rgb)              # left ↔ right RGB

        # -----------------------------------------------------------
        # 2-D vector overlays on the images
        # -----------------------------------------------------------
        def create_overlay(img_rgb, pm_xyz, motion_xyz, K,
                        do_subsample=True, subsample_factor=25,
                        cmap=cm.turbo):
            """
            returns copy of img_rgb with color-gradient lines.

            Args:
                img_rgb:     ndarray of shape (H, W, 3)
                pm_xyz:      ndarray of shape (H, W, 3), point map
                motion_xyz:  ndarray of shape (H, W, 3), motion map
                K:           camera intrinsics matrix (3×3)
                do_subsample:    bool, whether to subsample the points
                subsample_factor: int, grid size for subsampling
                cmap:        matplotlib colormap
            """
            h, w = img_rgb.shape[:2]
            overlay = img_rgb.copy()

            valid = (pm_xyz[:, :, 2] > 0) & (motion_xyz[:, :, 2] != 0)
            ys, xs = np.where(valid)

            if len(xs) == 0:
                return overlay  # (H, W, 3)

            # optionally subsample for clarity
            if do_subsample:
                grid_sz = subsample_factor ** 2
                step = max(1, len(xs) // (h * w // grid_sz))
                xs, ys = xs[::step], ys[::step]

            # start & end in cam coords
            start_xyz = pm_xyz[ys, xs]                  # (N, 3)
            end_xyz   = start_xyz + motion_xyz[ys, xs]  # (N, 3)

            # project → pixel coords
            start_uv = project_cam_pts(K, start_xyz)    # (N, 2)
            end_uv   = project_cam_pts(K, end_xyz)      # (N, 2)

            # keep only those fully inside image
            in_img = (
                (start_uv[:, 0] >= 0) & (start_uv[:, 0] < w) &
                (start_uv[:, 1] >= 0) & (start_uv[:, 1] < h) &
                (end_uv[:,   0] >= 0) & (end_uv[:,   0] < w) &
                (end_uv[:,   1] >= 0) & (end_uv[:,   1] < h)
            )

            for p0, p1 in zip(start_uv[in_img], end_uv[in_img]):
                draw_gradient_line(overlay, p0, p1, cmap=cmap, thick=1)

            return overlay  # (H, W, 3)


        # ---- left-camera overlays (left frame is reference) ------------
        ov_left_gt   = create_overlay(left_rgb,  left_pm_gt,  left_motion_gt,  K_L, cmap=cm.turbo)
        ov_left_pred = create_overlay(left_rgb,  left_pm_gt,  left_motion_pred, K_L, cmap=cm.turbo)

        # ---- right-camera overlays (right frame is reference) ----------
        #   need xyz in right-cam coords for projection; easiest: use pm already in R
        ov_right_gt  = create_overlay(right_rgb, right_pm_gt, right_motion_gt,  K_R, cmap=cm.turbo)
        ov_right_pred= create_overlay(right_rgb, right_pm_gt, right_motion_pred, K_R, cmap=cm.turbo)

        # ---- NEW overlays for left→right and right→left motions --------
        ov_l2r_gt   = create_overlay(left_rgb,  left_pm_gt,  l2r_motion_gt,  K_L, cmap=cm.turbo)
        ov_l2r_pred = create_overlay(left_rgb,  left_pm_gt,  l2r_motion_pred, K_L, cmap=cm.turbo)

        ov_r2l_gt   = create_overlay(right_rgb, right_pm_gt, r2l_motion_gt,  K_R, cmap=cm.turbo)
        ov_r2l_pred = create_overlay(right_rgb, right_pm_gt, r2l_motion_pred, K_R, cmap=cm.turbo)

        # -----------------------------------------------------------
        # log to wandb
        # -----------------------------------------------------------
        # hierarchical logging keys with confidence
        # base = f"e{epoch}_b{batch_idx}_{i}
        
        base = f"{base_name}_i{i}"
        # global k; base += f"_k{k}"; k+=1; print(base);

        log_dict = {
            # input images
            f"{base}/input/left"   : wandb.Image(left_rgb,  caption="left input"),
            f"{base}/input/mid"    : wandb.Image(mid_rgb,   caption="mid input"),
            f"{base}/input/right"  : wandb.Image(right_rgb, caption="right input"),

            # depth maps
            f"{base}/static/dm/left_gt"   : wandb.Image(hm_left_gt,   caption="left depth gt"),
            f"{base}/static/dm/left_pred" : wandb.Image(hm_left_pred, caption="left depth pred"),
            f"{base}/static/dm/right_gt"  : wandb.Image(hm_right_gt,  caption="right depth gt"),
            f"{base}/static/dm/right_pred": wandb.Image(hm_right_pred,caption="right depth pred"),

            # disparity maps
            f"{base}/static/disp/left_gt"   : wandb.Image(hm_disp_left_gt,   caption="left disparity gt"),
            f"{base}/static/disp/left_pred" : wandb.Image(hm_disp_left_pred, caption="left disparity pred"),
            f"{base}/static/disp/right_gt"  : wandb.Image(hm_disp_right_gt,  caption="right disparity gt"),
            f"{base}/static/disp/right_pred": wandb.Image(hm_disp_right_pred,caption="right disparity pred"),

            # 3d point clouds
            # f"{base}/static/pc/left_gt"   : wandb.Object3D(pc_left_gt),
            # f"{base}/static/pc/left_pred" : wandb.Object3D(pc_left_pred),
            # f"{base}/static/pc/right_gt"  : wandb.Object3D(pc_right_gt),
            # f"{base}/static/pc/right_pred": wandb.Object3D(pc_right_pred),

            # motion magnitude
            f"{base}/motion/magn/l2m_gt"   : wandb.Image(gray_motion_left_gt,   caption="left motion gt (magnitude)"),
            f"{base}/motion/magn/l2m_pred" : wandb.Image(gray_motion_left_pred, caption="left motion pred (magnitude)"),
            f"{base}/motion/magn/r2m_gt"   : wandb.Image(gray_motion_right_gt,  caption="right motion gt (magnitude)"),
            f"{base}/motion/magn/r2m_pred" : wandb.Image(gray_motion_right_pred,caption="right motion pred (magnitude)"),

            # motion magnitude (new directions)
            f"{base}/motion/magn/l2r_gt"  : wandb.Image(gray_motion_l2r_gt,  caption="left→right motion gt (mag)"),
            f"{base}/motion/magn/l2r_pred": wandb.Image(gray_motion_l2r_pred,caption="left→right motion pred (mag)"),
            f"{base}/motion/magn/r2l_gt"  : wandb.Image(gray_motion_r2l_gt,  caption="right→left motion gt (mag)"),
            f"{base}/motion/magn/r2l_pred": wandb.Image(gray_motion_r2l_pred,caption="right→left motion pred (mag)"),

            # 3d scene-flow visualizations
            f"{base}/motion/3d/l2m_gt"   : wandb.Image(viz_left_gt,   caption="3d scene flow left gt"),
            f"{base}/motion/3d/l2m_pred" : wandb.Image(viz_left_pred, caption="3d scene flow left pred"),
            f"{base}/motion/3d/r2m_gt"   : wandb.Image(viz_right_gt,  caption="3d scene flow right gt"),
            f"{base}/motion/3d/r2m_pred" : wandb.Image(viz_right_pred, caption="3d scene flow right pred"),
            # f"{base}/motion/3d_summary": wandb.Image(motion_3d_summary, caption="3D scene flow visualization"),

            # 3-D scene-flow images (new directions)
            f"{base}/motion/3d/l2r_gt"   : wandb.Image(viz_l2r_gt,  caption="3d scene flow left→right gt"),
            f"{base}/motion/3d/l2r_pred" : wandb.Image(viz_l2r_pred, caption="3d scene flow left→right pred"),
            f"{base}/motion/3d/r2l_gt"   : wandb.Image(viz_r2l_gt,  caption="3d scene flow right→left gt"),
            f"{base}/motion/3d/r2l_pred" : wandb.Image(viz_r2l_pred, caption="3d scene flow right→left pred"),

            # 2-d vector overlays
            f"{base}/motion/2d/left_gt"   : wandb.Image(ov_left_gt, caption="2d flow left gt"),
            f"{base}/motion/2d/left_pred" : wandb.Image(ov_left_pred, caption="2d flow left pred"),
            f"{base}/motion/2d/right_gt"  : wandb.Image(ov_right_gt, caption="2d flow right gt"),
            f"{base}/motion/2d/right_pred": wandb.Image(ov_right_pred, caption="2d flow right pred"),
            f"{base}/motion/2d/l2r_gt"    : wandb.Image(ov_l2r_gt,  caption="2d flow left→right gt"),
            f"{base}/motion/2d/l2r_pred"  : wandb.Image(ov_l2r_pred, caption="2d flow left→right pred"),
            f"{base}/motion/2d/r2l_gt"    : wandb.Image(ov_r2l_gt,  caption="2d flow right→left gt"),
            f"{base}/motion/2d/r2l_pred"  : wandb.Image(ov_r2l_pred, caption="2d flow right→left pred"),


            # 3-d scene-flow visualizations
            # f"{base}/scene-flow/3d_left_gt"   : wandb.Object3D(scene_left_gt),
            # f"{base}/scene-flow/3d_left_pred" : wandb.Object3D(scene_left_pred),
            # f"{base}/scene-flow/3d_right_gt"  : wandb.Object3D(scene_right_gt),
            # f"{base}/scene-flow/3d_right_pred": wandb.Object3D(scene_right_pred),

            # dm/img flip
            f"{base}/flip/left_input_gt_dm" : flip_left_gt,
            f"{base}/flip/left_input_pred_dm" : flip_left,
            f"{base}/flip/right_input_gt_dm": flip_right_gt,
            f"{base}/flip/right_input_pred_dm": flip_right,
            f"{base}/flip/left_depth_gt_pred": flip_left_depth_gt_pred,
            f"{base}/flip/right_depth_gt_pred": flip_right_depth_gt_pred,
            f"{base}/flip/left_right_rgb": flip_left_right_rgb,
        }

        # add confidence maps if available
        if gray_conf_left is not None:
            log_dict[f"{base}/static/conf/left_pred"] = wandb.Image(gray_conf_left, caption="left conf pred (1=black, inf=white)")
        if gray_conf_right is not None:
            log_dict[f"{base}/static/conf/right_pred"] = wandb.Image(gray_conf_right, caption="right conf pred (1=black, inf=white)")

        # add motion confidence maps if available
        if hm_motion_conf_left is not None:
            log_dict[f"{base}/motion/conf/left_pred"] = wandb.Image(hm_motion_conf_left, caption="left motion conf pred (1=black, inf=white)")
        if hm_motion_conf_right is not None:
            log_dict[f"{base}/motion/conf/right_pred"] = wandb.Image(hm_motion_conf_right, caption="right motion conf pred (1=black, inf=white)")
        if hm_motion_conf_l2r is not None:
            log_dict[f"{base}/motion/conf/l2r_pred"] = wandb.Image(hm_motion_conf_l2r, caption="left→right motion conf pred (1=black, inf=white)")
        if hm_motion_conf_r2l is not None:
            log_dict[f"{base}/motion/conf/r2l_pred"] = wandb.Image(hm_motion_conf_r2l, caption="right→left motion conf pred (1=black, inf=white)")

        wandb.log(log_dict, commit=True)


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

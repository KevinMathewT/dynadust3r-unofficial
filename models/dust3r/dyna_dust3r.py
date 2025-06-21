# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DynaDUSt3R model class - extends DUSt3R with motion prediction
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
import numpy as np
import matplotlib.cm as cm
from packaging import version
import huggingface_hub

import torch
import numpy as np
import wandb
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import io
from PIL import Image
import cv2
from time import time

from utils.geometry import normalize_pointcloud

from models.dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from models.dust3r.utils.heads import head_factory, motion_head_factory
from models.dust3r.utils.patch_embed import get_patch_embed

import loaders.utils.geometry as geom
from loaders.utils.viz import visualize_image, visualize_pm, visualize_sequence_from_pms

# import dust3r.utils.path_to_croco  # noqa: F401
from models.croco.croco import CroCoNet  # noqa

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
        teacher_forcing=True,  # Whether to use ground truth (True) or predicted (False) point clouds for motion
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.time_pos_emb_dim = time_pos_emb_dim
        self.teacher_forcing = teacher_forcing
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
        # print(f"image shape: {image.shape}")
        # print(f"image type: {image.dtype}")
        # print(f"true_shape: {true_shape}")
        # print(f"true_shape type: {true_shape.dtype}")
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
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def _motion_head(self, head_num, decout, img_shape, query_time):
        B, S, D = decout[-1].shape
        head = getattr(self, f"mhead{head_num}")
        return head(decout, img_shape, query_time)

    def forward(self, batch):
        """
        Forward pass for the DynaDUSt3R model.

        Parameters:
            batch: Dictionary containing at least 'left_image' and 'right_image' and optionally 'mid_tq'

        Returns:
            dict: Combined dictionary with results from both views
        """
        # extract views from batch
        left_view = {
            "img": batch["left_image"],  # (B, C, H, W)
            "true_shape": torch.tensor(batch["left_image"].shape[-2:])[None].repeat(
                batch["left_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch["left_instance"],  # [B x string]
        }
        right_view = {
            "img": batch["right_image"],  # (B, C, H, W)
            "true_shape": torch.tensor(batch["right_image"].shape[-2:])[None].repeat(
                batch["right_image"].size(0), 1
            ),  # (B, 2)
            "instance": batch["right_instance"],  # [B x string]
        }

        # get time query if available
        query_time = batch.get("mid_tq", None)  # (B,) if present

        # encode images
        (shape_left, shape_right), (feat_left, feat_right), (pos_left, pos_right) = (
            self._encode_symmetrized(left_view, right_view)
        )
        # shape_left, shape_right: each (B, 2) - image shape information
        # feat_left, feat_right: each (B, S, D) - tokenized features
        # pos_left, pos_right: each (B, S, D) - positional encodings

        # decode features
        dec_left, dec_right = self._decoder(feat_left, pos_left, feat_right, pos_right)
        # dec_left, dec_right: lists of tensors, each tensor (B, S, D) - features from multiple decoder layers

        with torch.amp.autocast(device_type=self.device_type, enabled=False):
            # get 3d points for both views
            res_left = self._downstream_head(
                1, [tok.float() for tok in dec_left], shape_left
            )
            # res_left: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)

            res_right = self._downstream_head(
                2, [tok.float() for tok in dec_right], shape_right
            )
            # res_right: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)

            # predict motion if time query provided
            if query_time is not None:
                motion_left = self._motion_head(
                    1, [tok.float() for tok in dec_left], shape_left, query_time
                )
                # motion_left: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)

                motion_right = self._motion_head(
                    2, [tok.float() for tok in dec_right], shape_right, query_time
                )
                # motion_right: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)

                # add motion to results with renamed keys to distinguish from point maps
                res_left["motion_map_pred"] = motion_left["map_pred"]  # (B, H, W, 3)
                res_right["motion_map_pred"] = motion_right["map_pred"]  # (B, H, W, 3)

                if "map_pred_conf" in motion_left:
                    res_left["motion_map_pred_conf"] = motion_left[
                        "map_pred_conf"
                    ]  # (B, H, W, 1)
                    res_right["motion_map_pred_conf"] = motion_right[
                        "map_pred_conf"
                    ]  # (B, H, W, 1)

        res_right["map_pred_in_left_frame"] = res_right.pop(
            "map_pred"
        )  # (B, H, W, 3) - right's pts3d in left's frame

        # combine results into single dictionary
        combined_results = {}

        # add left view results
        for k, v in res_left.items():
            combined_results[f"left_{k}"] = v

        # add right view results
        for k, v in res_right.items():
            combined_results[f"right_{k}"] = v

        # add batch size for metrics calculation
        combined_results["batch_size"] = batch["left_image"].size(0)

        return combined_results
        # single dict with keys:
        # 'left_map_pred': (B, H, W, 3) - 3D points from left view
        # 'left_map_pred_conf': optional (B, H, W, 1) - confidence for left view points
        # 'left_motion_map_pred': (B, H, W, 3) - motion vectors from left view, if query_time provided
        # 'left_motion_map_pred_conf': optional (B, H, W, 1) - confidence for left view motion
        # 'right_map_pred_in_left_frame': (B, H, W, 3) - 3D points from right view in left frame
        # 'right_map_pred_conf': optional (B, H, W, 1) - confidence for right view points
        # 'right_motion_map_pred': (B, H, W, 3) - motion vectors from right view, if query_time provided
        # 'right_motion_map_pred_conf': optional (B, H, W, 1) - confidence for right view motion
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

        return criterion(gt_left, gt_right, pred_left, pred_right)

    def get_motion_predictions(self, outputs, batch, device):
        """
        Add predicted motion to point maps and attach mid-frame validity.

        Args:
            outputs (dict):
                'left_map_pred': Tensor(B, H, W, 3),
                'left_motion_map_pred': Tensor(B, H, W, 3),
                'right_map_pred_in_left_frame': Tensor(B, H, W, 3),
                'right_motion_map_pred': Tensor(B, H, W, 3)
            batch (dict):
                'mid_pm': Tensor(B, H, W, 4),
                'left_pm': Tensor(B, H, W, 4),
                'right_pm': Tensor(B, H, W, 4)
            device (torch.device)
        Returns:
            pred_L (Tensor): (B, H, W, 4)
            pred_R (Tensor): (B, H, W, 4)
        """
        if self.teacher_forcing:
            # Use ground truth point clouds + predicted motion
            left_pts = batch["left_pm"][..., :3] + outputs["left_motion_map_pred"]  # (B, H, W, 3)
            right_pts = batch["right_pm"][..., :3] + outputs["right_motion_map_pred"]  # (B, H, W, 3)
            valid_L = batch["left_pm"][..., 3:].to(device)  # (B, H, W, 1)
            valid_R = batch["right_pm"][..., 3:].to(device)  # (B, H, W, 1)
        else:
            # Use predicted point clouds + predicted motion
            left_pts = outputs["left_map_pred"] + outputs["left_motion_map_pred"]  # (B, H, W, 3)
            right_pts = outputs["right_map_pred_in_left_frame"] + outputs["right_motion_map_pred"]  # (B, H, W, 3)
            # For validity, we need to compute based on predicted point clouds
            # Use predicted point cloud validity (assume valid where predictions exist)
            valid_L = torch.ones_like(outputs["left_map_pred"][..., :1]).to(device)  # (B, H, W, 1)
            valid_R = torch.ones_like(outputs["right_map_pred_in_left_frame"][..., :1]).to(device)  # (B, H, W, 1)
            
            # # If confidence maps are available, use them as validity
            # if "left_map_pred_conf" in outputs:
            #     valid_L = (outputs["left_map_pred_conf"] > 0.5).float()
            # if "right_map_pred_conf" in outputs:
            #     valid_R = (outputs["right_map_pred_conf"] > 0.5).float()
        
        pred_L = torch.cat([left_pts, valid_L], dim=-1)  # (B, H, W, 4)
        pred_R = torch.cat([right_pts, valid_R], dim=-1)  # (B, H, W, 4)
        return pred_L, pred_R

    def compute_motion_loss(self, criterion, batch, outputs, device):
        """
        Compute loss between motion-compensated preds and gt mid point map.

        Args:
            criterion (callable): fn(gt_L, gt_R, pred_L, pred_R) → scalar loss
            batch (dict): must contain 'mid_pm'
            outputs (dict): must contain motion preds
            device (torch.device)
        Returns:
            loss (Tensor) or (0.0, {}) on early exit
        """
        if (
            "left_motion_map_pred" not in outputs
            or "right_motion_map_pred" not in outputs
            or "mid_pm" not in batch
        ):
            raise ValueError(
                "Motion loss requires 'left_motion_map_pred', 'right_motion_map_pred', and 'mid_pm' in batch."
            )

        pred_L, pred_R = self.get_motion_predictions(
            outputs, batch, device
        )  # (B, H, W, 4), (B, H, W, 4)
        B = batch["mid_pm"].shape[0]  # ()
        eye = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B, 4, 4)

        gt_L = {
            "pts3d": batch["left_pm"][..., :3]
            + batch["left_to_mid_motion"][..., :3],  # (B, H, W, 3)
            "valid_mask": batch["left_to_mid_motion"][..., 3] > 0,  # (B, H, W)
            "camera_pose": eye,  # (B, 4, 4)
        }
        gt_R = {
            "pts3d": batch["right_pm"][..., :3]
            + batch["right_to_mid_motion"][..., :3],  # (B, H, W, 3)
            "valid_mask": batch["right_to_mid_motion"][..., 3] > 0,  # (B, H, W)
            "camera_pose": eye,  # (B, 4, 4)
        }

        pred_L_dict = {
            "pts3d": pred_L[..., :3],
            "valid_mask": pred_L[..., 3] > 0,
            "camera_pose": eye,
        }  # pts3d: (B, H, W, 3), valid_mask: (B, H, W), camera_pose: (B, 4, 4)
        pred_R_dict = {
            "pts3d": pred_R[..., :3],
            "valid_mask": pred_R[..., 3] > 0,
            "camera_pose": eye,
        }  # pts3d: (B, H, W, 3), valid_mask: (B, H, W), camera_pose: (B, 4, 4)

        if "left_motion_map_pred_conf" in outputs:
            pred_L_dict["conf"] = outputs["left_motion_map_pred_conf"]  # (B, H, W, 1)
        if "right_motion_map_pred_conf" in outputs:
            pred_R_dict["conf"] = outputs["right_motion_map_pred_conf"]  # (B, H, W, 1)
        loss = criterion(gt_L, gt_R, pred_L_dict, pred_R_dict)  # scalar
        return loss

    def get_loss(self, criterion, batch, outputs):
        """
        Compute total loss combining static reconstruction and motion prediction.

        Args:
            criterion: Loss criterion
            batch (dict): Batch data containing ground truth
            outputs (dict): Model outputs containing predictions

        Returns:
            tuple: (total_loss, loss_details)
        """
        device = batch["left_pm"].device

        # print out the sum of the valid masks for each i in the batch
        # for i in range(batch["left_image"].size(0)):
        #     print(f'get_loss | {i} | og mask1: {batch["left_pm"][i][..., 3].sum()} | og mask2: {batch["right_pm"][i][..., 3].sum()} | og shape: {batch["left_pm"][i].shape}')
        #     print(f'get_loss | {i} | instance1: {batch["left_instance"][i]} | instance2: {batch["right_instance"][i]}')

        static_loss, static_details = self.compute_static_loss(
            criterion, batch, outputs, device
        )
        motion_loss, motion_details = self.compute_motion_loss(
            criterion, batch, outputs, device
        )

        # Combine losses - with motion loss scaling factor if needed
        motion_weight = 1.0  # Adjust this value as needed (0.5, 0.1, etc.)
        total_loss = static_loss + motion_weight * motion_loss

        motion_details = {"motion_" + k: v for k, v in motion_details.items()}
        loss_details = {**static_details, **motion_details}
        loss_details["static_loss"] = (
            static_loss.item() if isinstance(static_loss, torch.Tensor) else static_loss
        )
        loss_details["motion_loss"] = (
            motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss
        )

        return total_loss, loss_details

    def compute_metrics(self, batch, outputs):
        """
        Compute evaluation metrics for model outputs.

        Parameters:
            batch: Dictionary with ground truth data
            outputs: Dictionary with model predictions

        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}

        # 3d point error for left view
        if "left_map_pred" in outputs and batch["left_pm"][..., 3].sum() > 0:
            mask = batch["left_pm"][..., 3] > 0
            pred_pts = outputs["left_map_pred"][mask]
            gt_pts = batch["left_pm"][..., :3][mask]

            dist = torch.norm(pred_pts - gt_pts, dim=-1)
            metrics["left_3d_error"] = dist.mean().item()

        # 3d point error for right view
        if (
            "right_map_pred_in_left_frame" in outputs
            and batch["right_pm"][..., 3].sum() > 0
        ):
            mask = batch["right_pm"][..., 3] > 0
            pred_pts = outputs["right_map_pred_in_left_frame"][mask]
            gt_pts = batch["right_pm"][..., :3][mask]

            dist = torch.norm(pred_pts - gt_pts, dim=-1)
            metrics["right_3d_error"] = dist.mean().item()

        # average 3d error
        if "left_3d_error" in metrics and "right_3d_error" in metrics:
            metrics["avg_3d_error"] = (
                metrics["left_3d_error"] + metrics["right_3d_error"]
            ) / 2

        # motion error for left view
        if (
            "left_motion_map_pred" in outputs
            and "left_to_mid_motion" in batch
            and batch["left_to_mid_motion"][..., 3].sum() > 0
        ):
            mask = batch["left_to_mid_motion"][..., 3] > 0
            pred_motion = outputs["left_motion_map_pred"][mask]
            gt_motion = batch["left_to_mid_motion"][..., :3][mask]

            dist = torch.norm(pred_motion - gt_motion, dim=-1)
            metrics["left_motion_error"] = dist.mean().item()

        # motion error for right view
        if (
            "right_motion_map_pred" in outputs
            and "right_to_mid_motion" in batch
            and batch["right_to_mid_motion"][..., 3].sum() > 0
        ):
            mask = batch["right_to_mid_motion"][..., 3] > 0
            pred_motion = outputs["right_motion_map_pred"][mask]
            gt_motion = batch["right_to_mid_motion"][..., :3][mask]

            dist = torch.norm(pred_motion - gt_motion, dim=-1)
            metrics["right_motion_error"] = dist.mean().item()

        # average motion error
        if "left_motion_error" in metrics and "right_motion_error" in metrics:
            metrics["avg_motion_error"] = (
                metrics["left_motion_error"] + metrics["right_motion_error"]
            ) / 2

        # add batch size (already included in forward method)

        return metrics


    def save_visualizations(self, batch, outputs, epoch, batch_idx, i=0, *args, **kwargs):
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

        def depth_to_heatmap(z):                  # (H,W) → (H,W,3) uint8
            valid = z > 0
            if not np.any(valid):                 # all invalid → black
                return np.zeros((*z.shape, 3), np.uint8)
            zv      = np.where(valid, z, np.nan)
            lo, hi  = np.nanmin(zv), np.nanmax(zv)
            norm    = (zv - lo) / (hi - lo + 1e-6)
            hm_rgb  = cm.turbo(norm)[:, :, :3]   # drop alpha, using turbo colormap
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
        # color-map depths
        # -----------------------------------------------------------
        hm_left_gt    = depth_to_heatmap(z_left_gt)
        hm_left_pred  = depth_to_heatmap(z_left_pred)
        hm_right_gt   = depth_to_heatmap(z_right_gt)
        hm_right_pred = depth_to_heatmap(z_right_pred)

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

        # -----------------------------------------------------------
        # motion maps (GT and predictions)
        # -----------------------------------------------------------
        # Extract GT motion vectors and validity
        left_motion_gt = batch["left_to_mid_motion"][i, :, :, :3].cpu().numpy() # (H,W,3)
        left_motion_validity = batch["left_to_mid_motion"][i, :, :, 3].cpu().numpy() # (H,W)
        
        right_motion_gt = batch["right_to_mid_motion"][i, :, :, :3].cpu().numpy() # (H,W,3)
        right_motion_validity = batch["right_to_mid_motion"][i, :, :, 3].cpu().numpy() # (H,W)
        
        # Extract predicted motion vectors
        left_motion_pred = outputs["left_motion_map_pred"][i, :, :, :].detach().cpu().numpy() # (H,W,3)
        right_motion_pred = outputs["right_motion_map_pred"][i, :, :, :].detach().cpu().numpy() # (H,W,3)

        # Scale motion predictions
        left_motion_pred = left_motion_pred * scale
        right_motion_pred = right_motion_pred * scale
        
        # Convert to grayscale magnitude visualizations
        gray_motion_left_gt = motion_magnitude_to_grayscale(left_motion_gt, left_motion_validity)
        gray_motion_right_gt = motion_magnitude_to_grayscale(right_motion_gt, right_motion_validity)
        gray_motion_left_pred = motion_magnitude_to_grayscale(left_motion_pred)
        gray_motion_right_pred = motion_magnitude_to_grayscale(right_motion_pred)

        # -----------------------------------------------------------
        # 3D motion field visualizations – four separate images
        # -----------------------------------------------------------
        viz_left_gt = visualize_3d_motion_field(
            left_pm_gt, left_motion_gt, left_rgb, left_motion_validity,
            subsample_factor=30, view_angles=(20, -45)
        )

        viz_left_pred = visualize_3d_motion_field(
            left_pm_gt, left_motion_pred, left_rgb, None,
            subsample_factor=30, view_angles=(20, -45)
        )

        viz_right_gt = visualize_3d_motion_field(
            right_pm_gt, right_motion_gt, right_rgb, right_motion_validity,
            subsample_factor=30, view_angles=(20, -135)
        )

        viz_right_pred = visualize_3d_motion_field(
            right_pm_gt, right_motion_pred, right_rgb, None,
            subsample_factor=30, view_angles=(20, -135)
        )
        # motion_3d_summary = create_motion_summary_figure(
        #     left_motion_gt, left_motion_pred,
        #     right_motion_gt, right_motion_pred,
        #     left_pm_gt, right_pm_gt,
        #     left_rgb, right_rgb,
        #     left_motion_validity, right_motion_validity
        # )

        # -----------------------------------------------------------
        # motion confidence maps (predictions only)
        # -----------------------------------------------------------
        conf_motion_left_pred = None
        conf_motion_right_pred = None
        if "left_motion_map_pred_conf" in outputs:
            conf_motion_left_pred = outputs["left_motion_map_pred_conf"][i, :, :].detach().cpu().numpy()
        if "right_motion_map_pred_conf" in outputs:
            conf_motion_right_pred = outputs["right_motion_map_pred_conf"][i, :, :].detach().cpu().numpy()
        
        hm_motion_conf_left = conf_to_grayscale(conf_motion_left_pred) if conf_motion_left_pred is not None else None
        hm_motion_conf_right = conf_to_grayscale(conf_motion_right_pred) if conf_motion_right_pred is not None else None

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
        def create_overlay(img_rgb, pm_xyz, motion_xyz, K, subsample=25,
                        cmap=cm.turbo):
            """returns copy of img_rgb with color-gradient lines."""
            H, W = img_rgb.shape[:2]
            overlay = img_rgb.copy()

            valid = (pm_xyz[:, :, 2] > 0) & (motion_xyz[:, :, 2] != 0)
            ys, xs = np.where(valid)

            if len(xs) == 0:
                return overlay

            # sparse sampling for clarity
            step = max(1, len(xs) // (H * W // (subsample ** 2)))
            xs, ys = xs[::step], ys[::step]

            # start & end xyz (cam frame of pm_xyz)
            start_xyz = pm_xyz[ys, xs]                                   # (N,3)
            end_xyz   = start_xyz + motion_xyz[ys, xs]                   # (N,3)

            # project → pixel coords
            start_uv = project_cam_pts(K, start_xyz)                     # (N,2)
            end_uv   = project_cam_pts(K, end_xyz)                       # (N,2)

            in_img = (
                (start_uv[:, 0] >= 0) & (start_uv[:, 0] < W) &
                (start_uv[:, 1] >= 0) & (start_uv[:, 1] < H) &
                (end_uv[:,   0] >= 0) & (end_uv[:,   0] < W) &
                (end_uv[:,   1] >= 0) & (end_uv[:,   1] < H)
            )
            for p0, p1 in zip(start_uv[in_img], end_uv[in_img]):
                draw_gradient_line(overlay, p0, p1, cmap=cmap, thick=1)

            return overlay                                             # (H,W,3)

        # ---- left-camera overlays ----
        ov_left_gt   = create_overlay(left_rgb,  left_pm_gt,  left_motion_gt,
                                    K_L, cmap=cm.turbo)
        ov_left_pred = create_overlay(left_rgb,  left_pm_gt,  left_motion_pred,
                                    K_L, cmap=cm.turbo)

        # ---- right-camera overlays ----
        #   need xyz in right-cam coords for projection
        right_pm_gt_camR   = geom.world_pc_to_cam_pc(
                                geom.cam_pc_to_world_pc(right_pm_gt.reshape(-1, 3), (K_R, E_R)),
                                (K_R, E_R)).reshape(*right_pm_gt.shape)   # (H,W,3)

        # but easier: reuse transform_pointmap_to_right_frame result
        ov_right_gt = create_overlay(right_rgb, right_pm_gt,
                                    right_motion_gt, K_R, cmap=cm.turbo)
        ov_right_pred = create_overlay(right_rgb, right_pm_gt,
                                    right_motion_pred, K_R, cmap=cm.turbo)

        # -----------------------------------------------------------
        # 3-D Object3D scene-flow overlays
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # 3-D Object3D scene-flow overlays (simple version)
        # -----------------------------------------------------------
        def build_scene_flow_object(pm,              # (H,W,4) or (H,W,3)
                                    motion,          # (H,W,3)
                                    motion_valid,    # (H,W) 1/0  or None
                                    rgb_img, K,
                                    max_points=300_000,
                                    n_seg=8, cmap=cm.turbo):
            """
            • point cloud: every GT-valid point (down-sampled only by max_points)
            • vectors   : only where pm-valid & motion-valid & motion ≠ 0
            """
            H, W = pm.shape[:2]

            # -------- point cloud --------------------------------------------------
            pc = pointmap_to_colored_pointcloud(pm, rgb_img, K, max_points)  # (N,6)
            if pc.size == 0:
                return {"type": "lidar/beta",
                        "points": pc.astype(np.float32),
                        "vectors": np.empty((0,), dtype=object)}

            # we need pixel coords corresponding to the pc rows
            # rebuild ys,xs the same way pointmap_to_colored_pointcloud scans
            valid_pm = pm[..., 2] > 0
            if pm.shape[-1] == 4:
                valid_pm &= pm[..., 3] > 0
            ys, xs = np.where(valid_pm)
            if len(ys) > max_points:
                idx = np.random.choice(len(ys), max_points, replace=False)
                ys, xs = ys[idx], xs[idx]

            # -------- vector start/end --------------------------------------------
            # motion validity mask
            mv = np.ones_like(motion[..., 0], bool) if motion_valid is None else (motion_valid > 0)
            vec_mask = valid_pm & mv & (np.linalg.norm(motion, axis=-1) > 0)

            ys_vec, xs_vec = np.where(vec_mask)
            if len(ys_vec) > max_points:
                idx = np.random.choice(len(ys_vec), max_points, replace=False)
                ys_vec, xs_vec = ys_vec[idx], xs_vec[idx]

            start_xyz = pm[ys_vec, xs_vec, :3]                      # (M,3)
            end_xyz   = start_xyz + motion[ys_vec, xs_vec]          # (M,3)

            # -------- vectors ---------------------------------------------------------
            vecs = []
            neon_green = [0, 255, 0]                     # B, G, R for W&B viewer
            for s, e in zip(start_xyz, end_xyz):
                if np.allclose(s, e):                    # skip zero motion
                    continue
                vecs.append({
                    "start": s.tolist(),
                    "end"  : e.tolist(),
                    "color": neon_green
                })


            return {
                "type": "lidar/beta",
                "points": pc.astype(np.float32),
                # "vectors": np.asarray(vecs, dtype=object)
            }


        # build four scenes (same sampling as 2-D overlay)
        scene_left_gt   = build_scene_flow_object(batch["left_pm"][i].cpu().numpy(),
                                                left_motion_gt,
                                                left_motion_validity,
                                                left_rgb, K_L)

        scene_left_pred = build_scene_flow_object(batch["left_pm"][i].cpu().numpy(),
                                                left_motion_pred,
                                                None,                  # no validity channel
                                                left_rgb, K_L)

        scene_right_gt  = build_scene_flow_object(right_pm_gt,
                                                right_motion_gt,
                                                right_motion_validity,
                                                right_rgb, K_R)

        scene_right_pred= build_scene_flow_object(right_pm_gt,
                                                right_motion_pred,
                                                None,
                                                right_rgb, K_R)

        # -----------------------------------------------------------
        # log to wandb
        # -----------------------------------------------------------
        # hierarchical logging keys with confidence
        # base = f"e{epoch}_b{batch_idx}_{i}
        base = f"e{epoch}_b{batch_idx}_i{i}"

        log_dict = {
            # input images
            f"{base}/input/left"   : wandb.Image(left_rgb,  caption="left input"),
            f"{base}/input/mid"    : wandb.Image(mid_rgb,   caption="mid input"),
            f"{base}/input/right"  : wandb.Image(right_rgb, caption="right input"),

            # depth maps
            f"{base}/static-dm/left_gt"   : wandb.Image(hm_left_gt,   caption="left depth gt"),
            f"{base}/static-dm/left_pred" : wandb.Image(hm_left_pred, caption="left depth pred"),
            f"{base}/static-dm/right_gt"  : wandb.Image(hm_right_gt,  caption="right depth gt"),
            f"{base}/static-dm/right_pred": wandb.Image(hm_right_pred,caption="right depth pred"),

            # motion magnitude
            f"{base}/motion/left_gt"   : wandb.Image(gray_motion_left_gt,   caption="left motion gt (magnitude)"),
            f"{base}/motion/left_pred" : wandb.Image(gray_motion_left_pred, caption="left motion pred (magnitude)"),
            f"{base}/motion/right_gt"  : wandb.Image(gray_motion_right_gt,  caption="right motion gt (magnitude)"),
            f"{base}/motion/right_pred": wandb.Image(gray_motion_right_pred,caption="right motion pred (magnitude)"),

            # 3d scene-flow visualizations
            f"{base}/scene-flow/3d_left_gt"   : wandb.Image(viz_left_gt,   caption="3d scene flow left gt"),
            f"{base}/scene-flow/3d_left_pred" : wandb.Image(viz_left_pred, caption="3d scene flow left pred"),
            f"{base}/scene-flow/3d_right_gt"  : wandb.Image(viz_right_gt,  caption="3d scene flow right gt"),
            f"{base}/scene-flow/3d_right_pred": wandb.Image(viz_right_pred, caption="3d scene flow right pred"),
            # f"{base}/motion/3d_summary": wandb.Image(motion_3d_summary, caption="3D scene flow visualization"),

            # 3d point clouds
            f"{base}/static-pc/left_gt"   : wandb.Object3D(pc_left_gt),
            f"{base}/static-pc/left_pred" : wandb.Object3D(pc_left_pred),
            f"{base}/static-pc/right_gt"  : wandb.Object3D(pc_right_gt),
            f"{base}/static-pc/right_pred": wandb.Object3D(pc_right_pred),

            # 2-d vector overlays
            f"{base}/motion/2d_left_gt"   : wandb.Image(ov_left_gt, caption="2-d flow left gt"),
            f"{base}/motion/2d_left_pred" : wandb.Image(ov_left_pred, caption="2-d flow left pred"),
            f"{base}/motion/2d_right_gt"  : wandb.Image(ov_right_gt, caption="2-d flow right gt"),
            f"{base}/motion/2d_right_pred": wandb.Image(ov_right_pred, caption="2-d flow right pred"),

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
            log_dict[f"{base}/static-conf/left_pred"] = wandb.Image(gray_conf_left, caption="left conf pred (1=black, inf=white)")
        if gray_conf_right is not None:
            log_dict[f"{base}/static-conf/right_pred"] = wandb.Image(gray_conf_right, caption="right conf pred (1=black, inf=white)")

        # add motion confidence maps if available
        if hm_motion_conf_left is not None:
            log_dict[f"{base}/motion-conf/left_pred"] = wandb.Image(hm_motion_conf_left, caption="left motion conf pred (1=black, inf=white)")
        if hm_motion_conf_right is not None:
            log_dict[f"{base}/motion-conf/right_pred"] = wandb.Image(hm_motion_conf_right, caption="right motion conf pred (1=black, inf=white)")

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

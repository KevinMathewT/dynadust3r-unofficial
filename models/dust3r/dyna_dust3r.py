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
from packaging import version
import huggingface_hub

from models.dust3r.utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from models.dust3r.utils.heads import head_factory, motion_head_factory
from models.dust3r.utils.patch_embed import get_patch_embed

import loaders.utils.geometry as geom
from loaders.utils.viz import (
    visualize_image, 
    visualize_pm, 
    visualize_sequence_from_pms
)

# import dust3r.utils.path_to_croco  # noqa: F401
from models.croco.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True):
    pass


class DynaDUSt3R(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d", "motion"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry), and to predict motion for given time queries.   
    """

    def __init__(self,
                 output_mode='pts3d',
                 motion_output_mode='pts3d',
                 head_type='linear',
                 motion_head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 time_pos_emb_dim=128,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.time_pos_emb_dim = time_pos_emb_dim
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, motion_output_mode, head_type, motion_head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        # random test stuff:
        self.k = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(DynaDUSt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, motion_output_mode, head_type, motion_head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.motion_output_mode = motion_output_mode
        self.head_type = head_type
        self.motion_head_type = motion_head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate point heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        
        # allocate motion heads
        self.motion_head1 = motion_head_factory(motion_head_type, motion_output_mode, self, has_conf=bool(conf_mode), time_pos_emb_dim=self.time_pos_emb_dim)
        self.motion_head2 = motion_head_factory(motion_head_type, motion_output_mode, self, has_conf=bool(conf_mode), time_pos_emb_dim=self.time_pos_emb_dim)
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
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

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
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)
    
    def _motion_head(self, head_num, decout, img_shape, query_time):
        B, S, D = decout[-1].shape
        head = getattr(self, f'mhead{head_num}')
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
            'img': batch['left_image'],  # (B, C, H, W)
            'true_shape': torch.tensor(
                batch['left_image'].shape[-2:])[None]
                    .repeat(batch['left_image'].size(0), 1),  # (B, 2)
            'instance': batch['left_instance'],  # [B x string]
        }
        right_view = {
            'img': batch['right_image'],  # (B, C, H, W)
            'true_shape': torch.tensor(
                batch['right_image'].shape[-2:])[None]
                    .repeat(batch['right_image'].size(0), 1),  # (B, 2)
            'instance': batch['right_instance'],  # [B x string]
        }
        
        # get time query if available
        query_time = batch.get('mid_tq', None)  # (B,) if present
        
        # encode images
        (shape_left, shape_right), (feat_left, feat_right), (pos_left, pos_right) = self._encode_symmetrized(left_view, right_view)
        # shape_left, shape_right: each (B, 2) - image shape information
        # feat_left, feat_right: each (B, S, D) - tokenized features
        # pos_left, pos_right: each (B, S, D) - positional encodings

        # decode features
        dec_left, dec_right = self._decoder(feat_left, pos_left, feat_right, pos_right)
        # dec_left, dec_right: lists of tensors, each tensor (B, S, D) - features from multiple decoder layers

        with torch.amp.autocast(device_type=self.device_type, enabled=False):
            # get 3d points for both views
            res_left = self._downstream_head(1, [tok.float() for tok in dec_left], shape_left)
            # res_left: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)
            
            res_right = self._downstream_head(2, [tok.float() for tok in dec_right], shape_right)
            # res_right: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)
            
            # predict motion if time query provided
            if query_time is not None:
                motion_left = self._motion_head(1, [tok.float() for tok in dec_left], shape_left, query_time)
                # motion_left: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)
                
                motion_right = self._motion_head(2, [tok.float() for tok in dec_right], shape_right, query_time)
                # motion_right: dict with 'map_pred': (B, H, W, 3), 'map_pred_conf': optional (B, H, W, 1)
                
                # add motion to results with renamed keys to distinguish from point maps
                res_left['motion_map_pred'] = motion_left['map_pred']  # (B, H, W, 3)
                res_right['motion_map_pred'] = motion_right['map_pred']  # (B, H, W, 3)
                
                if 'map_pred_conf' in motion_left:
                    res_left['motion_map_pred_conf'] = motion_left['map_pred_conf']  # (B, H, W, 1)
                    res_right['motion_map_pred_conf'] = motion_right['map_pred_conf']  # (B, H, W, 1)

        res_right['map_pred_in_left_frame'] = res_right.pop('map_pred')  # (B, H, W, 3) - right's pts3d in left's frame
        
        # combine results into single dictionary
        combined_results = {}
        
        # add left view results
        for k, v in res_left.items():
            combined_results[f'left_{k}'] = v
        
        # add right view results
        for k, v in res_right.items():
            combined_results[f'right_{k}'] = v
            
        # add batch size for metrics calculation
        combined_results['batch_size'] = batch['left_image'].size(0)
            
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
        batch_size = batch['left_image'].size(0)
        identity = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        gt_left = {
            'pts3d': batch['left_pm'][..., :3],
            'valid_mask': batch['left_pm'][..., 3] > 0,
            'camera_pose': identity
        }
        
        gt_right = {
            'pts3d': batch['right_pm'][..., :3],
            'valid_mask': batch['right_pm'][..., 3] > 0,
            'camera_pose': identity
        }
        
        pred_left = {'pts3d': outputs['left_map_pred']}
        pred_right = {'pts3d_in_other_view': outputs['right_map_pred_in_left_frame']}
        
        if 'left_map_pred_conf' in outputs:
            pred_left['conf'] = outputs['left_map_pred_conf']
        
        if 'right_map_pred_conf' in outputs:
            pred_right['conf'] = outputs['right_map_pred_conf']
        
        return criterion(gt_left, gt_right, pred_left, pred_right)


    def get_motion_predictions(self, outputs, batch, device):
        """
        Computes predicted 3D point maps (with visibility masks) from motion outputs by reprojecting motion-compensated
        predicted point maps using camera intrinsics and depth-based z-buffering.
        Args:
            outputs (dict): A dictionary containing predicted point maps and motion vectors.
                - 'left_map_pred': Tensor of shape (B, H, W, 3) — predicted 3D point map from the left view.
                - 'left_motion_map_pred': Tensor of shape (B, H, W, 3) — predicted motion vectors for left view.
                - 'right_map_pred_in_left_frame': Tensor of shape (B, H, W, 3) — predicted right-view point map aligned to left.
                - 'right_motion_map_pred': Tensor of shape (B, H, W, 3) — predicted motion vectors for right view.
            batch (dict): A dictionary containing input batch data.
                - 'left_pm': Tensor of shape (B, H, W, 4) — ground-truth left point map with validity in 4th channel.
                - 'mid_pm': Tensor of shape (B, H, W, 4) — mid frame point map with 4th channel indicating validity.
                - 'cam': List of tuples — each containing (intrinsics, extrinsics) for each sample in batch. Intrinsics is (3,3)/(4,4).
            device (torch.device): Device to place all computation on.
        Returns:
            pred_L (torch.Tensor): Reprojected 3D point map for left image after motion compensation, shape (B, H, W, 4).
                                Last channel is a binary mask indicating valid projected points.
            pred_R (torch.Tensor): Same as pred_L but for right image, reprojected in left frame.
        """
        batch_size = outputs['left_map_pred'].shape[0]
        height, width = outputs['left_map_pred'].shape[1:3]
        
        # Initialize output lists
        pred_L = []
        pred_R = []
        
        for b in range(batch_size):
            # Extract maps and motions for this batch
            # left_map = outputs['left_map_pred'][b]  # (H, W, 3)
            left_map = batch['left_pm'][b][..., :3]  # (H, W, 3)
            left_motion = outputs['left_motion_map_pred'][b]  # (H, W, 3)
            # right_map = outputs['right_map_pred_in_left_frame'][b]  # (H, W, 3)
            right_map = batch['right_pm'][b][..., :3]  # (H, W, 3)
            right_motion = outputs['right_motion_map_pred'][b]  # (H, W, 3)
            
            # Reshape to point clouds
            left_pc = left_map.reshape(-1, 3)  # (H*W, 3)
            left_motion_pc = left_motion.reshape(-1, 3)  # (H*W, 3)
            right_pc = right_map.reshape(-1, 3)  # (H*W, 3)
            right_motion_pc = right_motion.reshape(-1, 3)  # (H*W, 3)
            
            # Apply motion
            left_pc_moved = left_pc + left_motion_pc  # (H*W, 3)
            right_pc_moved = right_pc + right_motion_pc  # (H*W, 3)

            # Reproject using camera parameters
            camera = batch['cam'][b]
            left_pm = geom.cam_pc_to_cam_pm_with_torch(left_pc_moved, camera, (height, width))
            right_pm = geom.cam_pc_to_cam_pm_with_torch(right_pc_moved, camera, (height, width))
            
            pred_L.append(left_pm)
            pred_R.append(right_pm)

            # visualize
            if (b + self.k) % 2 == 0:
                if (b + self.k) % 4 == 0:
                    pms = [left_pc, left_pm]
                    motion_map = left_motion_pc.unsqueeze(0)
                    visualize_sequence_from_pms(
                        pms=pms,
                        motion_map=motion_map,
                        name="left_motion_in_batch_" + str(b)
                    )

                    pms = [left_pc, batch['mid_pm'][b][..., :3]]
                    motion_map = batch['left_to_mid_motion'][b][..., :3].unsqueeze(0)
                    visualize_sequence_from_pms(
                        pms=pms,
                        motion_map=motion_map,
                        name="gt_left_motion_in_batch_" + str(b)
                    )

                if (b + self.k) % 4 == 2:
                    pms = [right_pc, right_pm]
                    motion_map = right_motion_pc.unsqueeze(0)
                    visualize_sequence_from_pms(
                        pms=pms,
                        motion_map=motion_map,
                        name="right_motion_in_batch_" + str(b)
                    )

                    pms = [right_pc, batch['mid_pm'][b][..., :3]]
                    motion_map = batch['right_to_mid_motion'][b][..., :3].unsqueeze(0)
                    visualize_sequence_from_pms(
                        pms=pms,
                        motion_map=motion_map,
                        name="gt_right_motion_in_batch_" + str(b)
                    )

                self.k += 1
            
        
        # Stack results along batch dimension
        pred_L = torch.stack(pred_L, dim=0)
        pred_R = torch.stack(pred_R, dim=0)
        
        return pred_L, pred_R


    def compute_motion_loss(self, criterion, batch, outputs, device):
        if ('left_motion_map_pred' not in outputs or 
            'right_motion_map_pred' not in outputs or 
            'mid_pm' not in batch):
            return 0.0, {}  # ()

        pred_L, pred_R = self.get_motion_predictions(outputs, batch, device)  # (B, H, W, 3), (B, H, W, 3)
        B = batch['left_pm'].shape[0]  # (B)
        eye = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)  # (B, 4, 4)

        gt_L = {
            'pts3d': batch['mid_pm'][..., :3],  # (B, H, W, 3)
            'valid_mask': batch['mid_pm'][..., 3] > 0,  # (B, H, W)
            'camera_pose': eye  # (B, 4, 4)
        }
        gt_R = {
            'pts3d': batch['mid_pm'][..., :3],  # (B, H, W, 3)
            'valid_mask': batch['mid_pm'][..., 3] > 0,  # (B, H, W)
            'camera_pose': eye  # (B, 4, 4)
        }
        pred_l = {
            'pts3d': pred_L[..., :3],  # (B, H, W, 3)
            'valid_mask': pred_L[..., 3] > 0,  # (B, H, W)
            'camera_pose': eye  # (B, 4, 4)
        }
        pred_r = {
            'pts3d': pred_R[..., :3],  # (B, H, W, 3)
            'valid_mask': pred_R[..., 3] > 0,  # (B, H, W)
            'camera_pose': eye  # (B, 4, 4)
        }
        if 'left_motion_map_pred_conf' in outputs:
            pred_l['conf'] = outputs['left_motion_map_pred_conf']  # (B, H, W, 1)
        if 'right_motion_map_pred_conf' in outputs:
            pred_r['conf'] = outputs['right_motion_map_pred_conf']  # (B, H, W, 1)
        loss = criterion(gt_L, gt_R, pred_l, pred_r)  # scalar
        return loss  # scalar, dict



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
        device = batch['left_pm'].device
        
        static_loss, static_details = self.compute_static_loss(criterion, batch, outputs, device)
        motion_loss, motion_details = self.compute_motion_loss(criterion, batch, outputs, device)
        
        # Combine losses - with motion loss scaling factor if needed
        motion_weight = 0.0  # Adjust this value as needed (0.5, 0.1, etc.)
        total_loss = static_loss + motion_weight * motion_loss
        
        loss_details = {**static_details, **motion_details}
        loss_details['static_loss'] = static_loss.item()
        loss_details['motion_loss'] = motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss
        
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
        if 'left_map_pred' in outputs and batch['left_pm'][..., 3].sum() > 0:
            mask = batch['left_pm'][..., 3] > 0
            pred_pts = outputs['left_map_pred'][mask]
            gt_pts = batch['left_pm'][..., :3][mask]
            
            dist = torch.norm(pred_pts - gt_pts, dim=-1)
            metrics['left_3d_error'] = dist.mean().item()
        
        # 3d point error for right view
        if 'right_map_pred_in_left_frame' in outputs and batch['right_pm'][..., 3].sum() > 0:
            mask = batch['right_pm'][..., 3] > 0
            pred_pts = outputs['right_map_pred_in_left_frame'][mask]
            gt_pts = batch['right_pm'][..., :3][mask]
            
            dist = torch.norm(pred_pts - gt_pts, dim=-1)
            metrics['right_3d_error'] = dist.mean().item()
        
        # average 3d error
        if 'left_3d_error' in metrics and 'right_3d_error' in metrics:
            metrics['avg_3d_error'] = (metrics['left_3d_error'] + metrics['right_3d_error']) / 2
        
        # motion error for left view
        if 'left_motion_map_pred' in outputs and 'left_to_mid_motion' in batch and batch['left_to_mid_motion'][..., 3].sum() > 0:
            mask = batch['left_to_mid_motion'][..., 3] > 0
            pred_motion = outputs['left_motion_map_pred'][mask]
            gt_motion = batch['left_to_mid_motion'][..., :3][mask]
            
            dist = torch.norm(pred_motion - gt_motion, dim=-1)
            metrics['left_motion_error'] = dist.mean().item()
        
        # motion error for right view
        if 'right_motion_map_pred' in outputs and 'right_to_mid_motion' in batch and batch['right_to_mid_motion'][..., 3].sum() > 0:
            mask = batch['right_to_mid_motion'][..., 3] > 0
            pred_motion = outputs['right_motion_map_pred'][mask]
            gt_motion = batch['right_to_mid_motion'][..., :3][mask]
            
            dist = torch.norm(pred_motion - gt_motion, dim=-1)
            metrics['right_motion_error'] = dist.mean().item()
        
        # average motion error
        if 'left_motion_error' in metrics and 'right_motion_error' in metrics:
            metrics['avg_motion_error'] = (metrics['left_motion_error'] + metrics['right_motion_error']) / 2
        
        # add batch size (already included in forward method)
        
        return metrics

    def save_visualizations(self, batch, outputs, epoch, batch_idx, path):
        import os
        import numpy as np

        if batch['left_pm'].shape[0] == 0:
            return

        i = 0
        device = batch['left_pm'].device

        out_dir = os.path.join(path, f"viz_e{epoch}_b{batch_idx}")
        os.makedirs(out_dir, exist_ok=True)

        left_img  = batch['left_image'][i].detach().cpu().permute(1,2,0).numpy()
        mid_img   = batch['mid_image'][i].detach().cpu().permute(1,2,0).numpy()
        right_img = batch['right_image'][i].detach().cpu().permute(1,2,0).numpy()

        visualize_image(left_img,  name="left_image")
        visualize_image(mid_img,   name="mid_image")
        visualize_image(right_img, name="right_image")

        left_pm  = batch['left_pm'][i].detach().cpu().numpy()
        mid_pm   = batch['mid_pm'][i].detach().cpu().numpy()
        right_pm = batch['right_pm'][i].detach().cpu().numpy()

        # CHANGE ONLY THIS LINE:
        cam = batch['cam'][i].detach().cpu().numpy() if 'cam' in batch else None

        visualize_pm(left_pm,  image=left_img,  cam=cam, name="left_pm")
        visualize_pm(mid_pm,   image=None,   cam=cam, name="mid_pm")
        visualize_pm(right_pm, image=None, cam=cam, name="right_pm")

        left_to_mid  = batch['left_to_mid_motion'][i].detach().cpu().numpy()
        right_to_mid = batch['right_to_mid_motion'][i].detach().cpu().numpy()

        tracks_left_gt  = np.stack([left_pm, mid_pm], axis=0)
        tracks_right_gt = np.stack([right_pm, mid_pm], axis=0)
        motion_left_gt  = left_to_mid[np.newaxis, ...]
        motion_right_gt = right_to_mid[np.newaxis, ...]

        images_left_gt  = [left_img, None]
        images_right_gt = [None, None]

        visualize_sequence_from_pms(
            pms=tracks_left_gt,
            motion_map=motion_left_gt,
            image_seq=images_left_gt,
            name="tracks_left_gt"
        )
        visualize_sequence_from_pms(
            pms=tracks_right_gt,
            motion_map=motion_right_gt,
            image_seq=images_right_gt,
            name="tracks_right_gt"
        )

        pred_L, pred_R = self.get_motion_predictions(outputs, batch, device)
        midL = pred_L[i].detach().cpu().numpy()
        midR = pred_R[i].detach().cpu().numpy()

        left_motion_3d  = outputs['left_motion_map_pred'][i].detach().cpu().numpy()
        right_motion_3d = outputs['right_motion_map_pred'][i].detach().cpu().numpy()

        tracks_left_pred  = np.stack([left_pm, midL], axis=0)
        tracks_right_pred = np.stack([right_pm, midR], axis=0)

        motion_left_pred  = np.zeros((1, *left_pm.shape), dtype=left_pm.dtype)
        motion_right_pred = np.zeros((1, *right_pm.shape), dtype=right_pm.dtype)

        motion_left_pred[0, ..., :3]  = left_motion_3d
        motion_right_pred[0, ..., :3] = right_motion_3d

        mask_0 = (tracks_left_pred[0, ..., 3] > 0)
        mask_1 = (tracks_left_pred[1, ..., 3] > 0)
        motion_left_pred[0, ..., 3] = (mask_0 & mask_1).astype(np.float32)

        mask_0 = (tracks_right_pred[0, ..., 3] > 0)
        mask_1 = (tracks_right_pred[1, ..., 3] > 0)
        motion_right_pred[0, ..., 3] = (mask_0 & mask_1).astype(np.float32)

        images_left_pred  = [left_img, None]
        images_right_pred = [right_img, None]

        visualize_sequence_from_pms(
            pms=tracks_left_pred,
            motion_map=motion_left_pred,
            image_seq=images_left_pred,
            name="tracks_left_pred"
        )
        visualize_sequence_from_pms(
            pms=tracks_right_pred,
            motion_map=motion_right_pred,
            image_seq=images_right_pred,
            name="tracks_right_pred"
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
            best_filename = filename.replace('checkpoint_', 'model_best_')
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
        model_cfg.pop('name', None)
        use_pretrained = model_cfg.pop('use_pretrained', False)
        pretrained_url = model_cfg.pop('pretrained_link', None)

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
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                sd = checkpoint['model']
            else:
                sd = checkpoint

            # Copy DUSt3R point-head keys into motion-head keys
            for k, v in list(sd.items()):
                if k.startswith('head1'):
                    sd[k.replace('head1', 'mhead1')] = v.clone()
                elif k.startswith('head2'):
                    sd[k.replace('head2', 'mhead2')] = v.clone()

            # Load into the DynaDUSt3R model
            model.load_state_dict(sd, strict=False)
            print(f"loaded pretrained weights successfully from {save_path}.")

        return model


if __name__ == "__main__":
    # test model instantiation
    model = DynaDUSt3R()
    print(model)
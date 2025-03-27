# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DynaDUSt3R model class - extends DUSt3R with motion prediction
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from models.dust3r.utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from models.dust3r.utils.heads import head_factory, motion_head_factory
from models.dust3r.utils.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from croco.models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


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
                 head_type='linear',
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
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

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

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate point heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        
        # allocate motion heads
        self.motion_head1 = motion_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), time_pos_emb_dim=self.time_pos_emb_dim)
        self.motion_head2 = motion_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), time_pos_emb_dim=self.time_pos_emb_dim)
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
            'true_shape': batch['left_image'].shape[-2:]  # (H, W)
        }
        right_view = {
            'img': batch['right_image'],  # (B, C, H, W)
            'true_shape': batch['right_image'].shape[-2:]  # (H, W)
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

        with torch.amp.autocast(enabled=False):
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
    
    def get_loss(self, criterion, batch, outputs):
        """
        Compute loss between model outputs and ground truth.
        
        Parameters:
            criterion: Loss function
            batch: Dictionary with ground truth data
            outputs: Dictionary with model predictions
            
        Returns:
            tuple: (loss, loss_details)
        """
        # prepare ground truth data
        gt_left = {
            'pts3d': batch['left_pm'][..., :3],  # (B, H, W, 3)
            'valid_mask': batch['left_pm'][..., 3] > 0,  # (B, H, W)
            'camera_pose': torch.eye(4).unsqueeze(0).repeat(batch['left_image'].size(0), 1, 1).to(batch['left_image'].device)
        }
        
        gt_right = {
            'pts3d': batch['right_pm'][..., :3],  # (B, H, W, 3)
            'valid_mask': batch['right_pm'][..., 3] > 0,  # (B, H, W)
            'camera_pose': torch.eye(4).unsqueeze(0).repeat(batch['right_image'].size(0), 1, 1).to(batch['right_image'].device)
        }
        
        # prepare prediction data
        pred_left = {
            'pts3d': outputs['left_map_pred'],  # (B, H, W, 3)
        }
        
        pred_right = {
            'pts3d_in_other_view': outputs['right_map_pred_in_left_frame'],  # (B, H, W, 3)
        }
        
        # add confidence if available
        if 'left_map_pred_conf' in outputs:
            pred_left['conf'] = outputs['left_map_pred_conf']
        
        if 'right_map_pred_conf' in outputs:
            pred_right['conf'] = outputs['right_map_pred_conf']
        
        # compute loss using criterion
        loss, loss_details = criterion(gt_left, gt_right, pred_left, pred_right)
        
        return loss, loss_details

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


if __name__ == "__main__":
    # test model instantiation
    model = DynaDUSt3R()
    print(model)
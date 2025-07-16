# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Corrected DUSt3R training losses with unified normalization
# --------------------------------------------------------
import torch
import torch.nn as nn
from copy import copy

from utils.geometry import normalize_pointcloud


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class L21Loss(BaseCriterion):
    """Euclidean distance between 3d points"""
    
    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        dist = torch.norm(a - b, dim=-1)
        
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')


class UnifiedCriterion(nn.Module):
    """
    Unified criterion that computes total loss as Conf + Reg3d with single normalization.
    
    Key fix: Calculate normalization once using point head loss logic, then reuse 
    the same normalization for both point and motion map loss calculations.
    """
    
    def __init__(self, alpha=1.0, norm_mode='avg_dis', gt_scale=False):
        super().__init__()
        self.alpha = alpha  # confidence loss weight
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.base_criterion = L21Loss(reduction='none')
    
    def get_all_pts3d_with_normalization(self, gt1, gt2, pred1, pred2, dist_clip=None):
        """
        Extract and normalize point clouds. This is the SINGLE normalization calculation
        that should be reused for both point and motion losses.
        """
        # Get ground truth and predicted points directly (no camera transformation needed)
        gt_pts1 = gt1['pts3d']
        gt_pts2 = gt2['pts3d']
        
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()
        
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)
            dis2 = gt_pts2.norm(dim=-1)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        
        # Get predicted points directly from the predictions
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d_in_other_view'] if 'pts3d_in_other_view' in pred2 else pred2['pts3d']
        
        # CRITICAL: Extract BOTH normalization factors like original but save them for reuse
        pred_norm_factor = None
        gt_norm_factor = None
        
        if self.norm_mode:
            # Debug: Print point cloud stats before normalization
            # print(f"[DEBUG] BEFORE norm - pred_pts mean magnitude: {torch.cat([pr_pts1[valid1], pr_pts2[valid2]]).norm(dim=-1).mean().item():.6f}")
            # print(f"[DEBUG] BEFORE norm - gt_pts mean magnitude: {torch.cat([gt_pts1[valid1], gt_pts2[valid2]]).norm(dim=-1).mean().item():.6f}")
            
            # First normalize predicted points (like line 179 in original)
            pr_pts1_norm, pr_pts2_norm, pred_norm_factor = normalize_pointcloud(
                pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, ret_factor=True
            )
            
            # Then normalize ground truth points (like line 181 in original)
            if not self.gt_scale:
                gt_pts1_norm, gt_pts2_norm, gt_norm_factor = normalize_pointcloud(
                    gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True
                )
                
                # Debug: print normalization factors
                # print(f"[DEBUG] pred_norm_factor: {pred_norm_factor.item():.6f}")
                # print(f"[DEBUG] gt_norm_factor: {gt_norm_factor.item():.6f}")
                # print(f"[DEBUG] ratio (pred/gt): {(pred_norm_factor / gt_norm_factor).item():.6f}")
            else:
                gt_pts1_norm = gt_pts1
                gt_pts2_norm = gt_pts2
                gt_norm_factor = None
        else:
            pr_pts1_norm = pr_pts1
            pr_pts2_norm = pr_pts2
            gt_pts1_norm = gt_pts1
            gt_pts2_norm = gt_pts2
            
        # Return both normalization factors for reuse
        normalization_factors = (pred_norm_factor, gt_norm_factor)
        return gt_pts1_norm, gt_pts2_norm, pr_pts1_norm, pr_pts2_norm, valid1, valid2, normalization_factors
    
    def apply_normalization_factors(self, pred_pts1, pred_pts2, gt_pts1, gt_pts2, normalization_factors):
        """Apply pre-computed normalization factors to point clouds"""
        pred_norm_factor, gt_norm_factor = normalization_factors
        
        # Apply predicted normalization factor
        if pred_norm_factor is not None:
            pred_pts1 = pred_pts1 / pred_norm_factor
            pred_pts2 = pred_pts2 / pred_norm_factor
        
        # Apply ground truth normalization factor
        if gt_norm_factor is not None:
            gt_pts1 = gt_pts1 / gt_norm_factor
            gt_pts2 = gt_pts2 / gt_norm_factor
        
        return pred_pts1, pred_pts2, gt_pts1, gt_pts2
    
    def compute_regression_loss(self, gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2):
        """Compute L2 regression loss"""
        l1 = self.base_criterion(pred_pts1[mask1], gt_pts1[mask1])
        l2 = self.base_criterion(pred_pts2[mask2], gt_pts2[mask2])
        
        # Average losses
        l1_mean = l1.mean() if l1.numel() > 0 else torch.zeros((), device=l1.device)
        l2_mean = l2.mean() if l2.numel() > 0 else torch.zeros((), device=l1.device)
        
        return l1_mean + l2_mean, l1, l2
    
    def compute_confidence_loss(self, pixel_loss1, pixel_loss2, pred1, pred2, mask1, mask2):
        """Compute confidence-weighted loss"""
        if 'conf' not in pred1 or 'conf' not in pred2:
            return torch.zeros((), device=pixel_loss1.device)
        
        conf1 = pred1['conf'][mask1]
        conf2 = pred2['conf'][mask2]
        
        # Confidence loss: loss * conf - alpha * log(conf)
        conf_loss1 = pixel_loss1 * conf1 - self.alpha * torch.log(conf1)
        conf_loss2 = pixel_loss2 * conf2 - self.alpha * torch.log(conf2)
        
        conf_loss1_mean = conf_loss1.mean() if conf_loss1.numel() > 0 else torch.zeros((), device=conf_loss1.device)
        conf_loss2_mean = conf_loss2.mean() if conf_loss2.numel() > 0 else torch.zeros((), device=conf_loss2.device)
        
        return conf_loss1_mean + conf_loss2_mean
    
    def forward(self, gt1, gt2, pred1, pred2, normalization_factors=None, **kwargs):
        """
        Compute total loss as Conf + Reg3d with unified normalization.
        
        Args:
            gt1, gt2: Ground truth point clouds and metadata
            pred1, pred2: Predicted point clouds and metadata  
            normalization_factors: Tuple of (pred_norm_factor, gt_norm_factor) (if None, compute them)
            
        Returns:
            total_loss: Combined confidence + regression loss
            details: Loss breakdown dictionary
            normalization_factors: Tuple of normalization factors for reuse
        """
        if normalization_factors is None:
            # Compute normalization factors if not provided
            gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, norm_factors = \
                self.get_all_pts3d_with_normalization(gt1, gt2, pred1, pred2, **kwargs)
        else:
            # Use provided normalization factors
            gt_pts1 = gt1['pts3d']
            gt_pts2 = gt2['pts3d']
            
            pred_pts1 = pred1['pts3d']
            pred_pts2 = pred2['pts3d_in_other_view'] if 'pts3d_in_other_view' in pred2 else pred2['pts3d']
            
            # Apply the provided normalization factors
            if self.norm_mode:
                pred_pts1, pred_pts2, gt_pts1, gt_pts2 = self.apply_normalization_factors(
                    pred_pts1, pred_pts2, gt_pts1, gt_pts2, normalization_factors
                )
            
            mask1 = gt1['valid_mask']
            mask2 = gt2['valid_mask']
            norm_factors = normalization_factors
        
        # Compute regression loss
        reg_loss, pixel_loss1, pixel_loss2 = self.compute_regression_loss(
            gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2
        )
        
        # Compute confidence loss
        conf_loss = self.compute_confidence_loss(
            pixel_loss1, pixel_loss2, pred1, pred2, mask1, mask2
        )
        # conf_loss = torch.zeros_like(reg_loss) # should be removed in final version
        
        # Total loss = Conf + Reg3d
        total_loss = conf_loss + reg_loss
        
        # Debug: Print loss breakdown
        # print(f"[DEBUG] reg_loss: {reg_loss.item():.8f}")
        reg_loss_1_val = pixel_loss1.mean().item() if pixel_loss1.numel() > 0 else 0
        reg_loss_2_val = pixel_loss2.mean().item() if pixel_loss2.numel() > 0 else 0
        # print(f"[DEBUG] reg_loss_1: {reg_loss_1_val:.8f}")
        # print(f"[DEBUG] reg_loss_2: {reg_loss_2_val:.8f}")
        # print(f"[DEBUG] valid points mask1: {mask1.sum().item()}, mask2: {mask2.sum().item()}")
        
        # Detailed breakdown
        details = {
            'total_loss': float(total_loss),
            'conf_loss': float(conf_loss),
            'reg_loss': float(reg_loss),
            'reg_loss_1': float(pixel_loss1.mean() if pixel_loss1.numel() > 0 else 0),
            'reg_loss_2': float(pixel_loss2.mean() if pixel_loss2.numel() > 0 else 0),
        }
        
        return total_loss, details, norm_factors
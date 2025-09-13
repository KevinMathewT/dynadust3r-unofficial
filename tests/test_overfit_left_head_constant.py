#!/usr/bin/env python
"""
Test to overfit the left head of DynaDUSt3R to match constant outputs.

This test creates:
1. A constant input batch (left_image, right_image) 
2. A constant target output for the left head
3. A simple training loop that overfits only the left head output to match the target

Usage: poetry run python -m tests.test_overfit_left_head_constant
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra

# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_model
from optimizer import get_optimizer
from models.dust3r.utils.heads.postprocess import reg_dense_depth


def get_loss(model, batch, outputs):
    """
    Compute total loss with aggressive memory optimization.
    Fixed to match unoptimized version behavior exactly.

    Args:
        model: The model instance (for accessing depth_post_mode, etc.)
        batch (dict):
            - left_pm: (B, H, W, 4)
            - right_pm: (B, H, W, 4)
            - motion_gt (dict, optional):
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
            - motion_pred (dict, optional):
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
    left_pred_pp = reg_dense_depth(outputs["left_map_pred"], model.depth_post_mode)
    right_pred_pp = reg_dense_depth(outputs["right_map_pred_in_left_frame"], model.depth_post_mode)

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
        return reg_dense_depth(base_pc + motion_disp[..., :3], model.motion_depth_post_mode)

    # Build motion-postprocessed predictions
    pred_l2m_pp = _pp_motion_sum(outputs["left_map_pred"], outputs["motion_pred"][l2m_key])
    pred_r2m_pp = _pp_motion_sum(outputs["right_map_pred_in_left_frame"], outputs["motion_pred"][r2m_key])
    pred_l2r_pp = _pp_motion_sum(outputs["left_map_pred"], outputs["motion_pred"][l2r_key])
    pred_r2l_pp = _pp_motion_sum(outputs["right_map_pred_in_left_frame"], outputs["motion_pred"][r2l_key])

    # Process everything in chunks to avoid large intermediate tensors
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)  # ()
    loss_details = {}  # (dict)

    # Build loss configurations - comment out any you don't want to include
    loss_configs = [
        {
            "name": "left",
            "gt": batch["left_pm"][..., :3],  # (B, H, W, 3)
            "pred": left_pred_pp,  # (B, H, W, 3) postprocessed
            "valid": batch["left_pm"][..., 3] > 0,  # (B, H, W)
            "conf": outputs.get("left_map_pred_conf", None),  # (B, H, W, 1) or None
            "is_base": True
        },
        # {
        #     "name": "right",
        #     "gt": batch["right_pm"][..., :3],  # (B, H, W, 3)
        #     "pred": right_pred_pp,  # (B, H, W, 3) postprocessed
        #     "valid": batch["right_pm"][..., 3] > 0,  # (B, H, W)
        #     "conf": outputs.get("right_map_pred_conf", None),  # (B, H, W, 1) or None
        #     "is_base": True
        # },
        # {
        #     "name": "l2m",
        #     "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2m"][..., :3],
        #     "pred": pred_l2m_pp,
        #     "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2m"][..., 3] > 0),
        #     "conf": outputs["motion_pred"].get(f"l_to_t{tq_mid_idx}_conf", None),
        #     "is_base": False
        # },
        # {
        #     "name": "r2m",
        #     "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2m"][..., :3],
        #     "pred": pred_r2m_pp,
        #     "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2m"][..., 3] > 0),
        #     "conf": outputs["motion_pred"].get(f"r_to_t{tq_mid_idx}_conf", None),
        #     "is_base": False
        # },
        # {
        #     "name": "l2r",
        #     "gt": batch["left_pm"][..., :3] + batch["motion_gt"]["l2r"][..., :3],
        #     "pred": pred_l2r_pp,
        #     "valid": (batch["left_pm"][..., 3] > 0) & (batch["motion_gt"]["l2r"][..., 3] > 0),
        #     "conf": outputs["motion_pred"].get("l_to_r_conf", None),
        #     "is_base": False
        # },
        # {
        #     "name": "r2l",
        #     "gt": batch["right_pm"][..., :3] + batch["motion_gt"]["r2l"][..., :3],
        #     "pred": pred_r2l_pp,
        #     "valid": (batch["right_pm"][..., 3] > 0) & (batch["motion_gt"]["r2l"][..., 3] > 0),
        #     "conf": outputs["motion_pred"].get("r_to_l_conf", None),
        #     "is_base": False
        # }
    ]

    # Compute normalization factors once for base PCs
    with torch.no_grad():  # ()
        # Extract base PCs and validity
        gt_left_pc = batch["left_pm"][..., :3]  # (B, H, W, 3)
        gt_right_pc = batch["right_pm"][..., :3]  # (B, H, W, 3)
        valid_left = batch["left_pm"][..., 3] > 0  # (B, H, W)
        valid_right = batch["right_pm"][..., 3] > 0  # (B, H, W)

        # GT normalization - using the FIXED version to match unoptimized
        gt_scale = _compute_norm_factor_fixed(  # (B,1,1,1)
            gt_left_pc, gt_right_pc, valid_left, valid_right
        )

    # Pred normalization (needs gradients) — use postprocessed base PCs
    pred_scale = _compute_norm_factor_fixed(  # (B,1,1,1)
        left_pred_pp,
        right_pred_pp,
        valid_left, valid_right
    )

    gt_scale = 1
    pred_scale = 1

    # Process each loss component
    for cfg in loss_configs:  # ()
        # # debug: print shapes per component once
        # if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        #     def s(x): return None if x is None else tuple(x.shape)
        #     print(f"[loss-cfg] {cfg['name']}: gt={s(cfg['gt'])} pred={s(cfg['pred'])} valid={s(cfg['valid'])} conf={s(cfg['conf'])}")
        #     if cfg["conf"] is not None and cfg["conf"].ndim == 4 and cfg["conf"].shape[-1] != 1:
        #         print(f"[conf-multi] {cfg['name']} conf has C={cfg['conf'].shape[-1]} channels")  # (B, H, W, C)

        # Compute single loss component
        loss_comp, comp_stats = _compute_single_loss_fixed(  # ()
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


def _compute_norm_factor_fixed(pc1, pc2, valid1, valid2):
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


def _compute_single_loss_fixed(gt_pc, pred_pc, valid_mask, conf, 
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


def create_constant_batch(batch_size=2, height=None, width=None, config=None, device='cuda'):
    """Create a constant input batch with left and right images."""
    # Use config's image size if not specified
    if height is None and config is not None:
        height = config.model.img_size[0]
    if width is None and config is not None:
        width = config.model.img_size[1]
    if height is None:
        height = 256  # fallback
    if width is None:
        width = 256   # fallback

    # Constant input images (random but fixed)
    torch.manual_seed(42)  # Fixed seed for reproducible constants
    left_image = torch.randn(batch_size, 3, height, width, device=device) * 0.5 + 0.5
    right_image = torch.randn(batch_size, 3, height, width, device=device) * 0.5 + 0.5

    batch = {
        'left_image': left_image,
        'right_image': right_image,
        'batch_size': batch_size,
        'left_instance': ['test_left'] * batch_size,  # Instance info as list
        'right_instance': ['test_right'] * batch_size,  # Instance info as list
        'query_times': None,  # No motion queries for this test
    }
    return batch


def create_target_left_output(batch_size=2, height=None, width=None, config=None, device='cuda'):
    """Create a constant target output for the left head (3D points)."""
    # Use config's image size if not specified
    if height is None and config is not None:
        height = config.model.img_size[0]
    if width is None and config is not None:
        width = config.model.img_size[1]
    if height is None:
        height = 256  # fallback
    if width is None:
        width = 256   # fallback

    torch.manual_seed(123)  # Different seed for target
    # Target: (B, H, W, 3) representing 3D points (smaller values for easier overfitting)
    target_left_output = torch.randn(batch_size, height, width, 3, device=device) * 0.1
    return target_left_output


def compute_left_head_loss(predicted_left, target_left):
    """Compute MSE loss between predicted and target left head output."""
    return F.mse_loss(predicted_left, target_left)


def create_synthetic_pointmaps_and_motion(batch, config=None, device='cuda'):
    """Augment batch with synthetic pointmaps and motion GT compatible with get_loss.

    Adds keys:
      - left_pm:  (B, H, W, 4)  (xyz + valid)
      - right_pm: (B, H, W, 4)
      - motion_gt: dict with l2m, r2m, l2r, r2l each (B, H, W, 4)
      - query_times: (B, T) where index 0 is mid-time (e.g., 0.5), and times include 1.0 and 0.0
    """
    B = batch['left_image'].size(0)
    if config is not None:
        H, W = config.model.img_size[0], config.model.img_size[1]
    else:
        H = batch['left_image'].shape[-2]
        W = batch['left_image'].shape[-1]

    torch.manual_seed(777)
    # Base left pointmap (small magnitude for easy fitting)
    left_xyz = torch.randn(B, H, W, 3, device=device) * 0.1
    # Define a consistent displacement from left->right
    l2r_xyz = torch.randn(B, H, W, 3, device=device) * 0.02

    # Derive other components consistently
    right_xyz = left_xyz + l2r_xyz
    r2l_xyz = -l2r_xyz
    l2m_xyz = 0.5 * l2r_xyz
    r2m_xyz = -0.5 * l2r_xyz

    # Valid masks (all valid)
    v_left = torch.ones(B, H, W, 1, device=device)
    v_right = torch.ones(B, H, W, 1, device=device)
    v_l2r = torch.ones(B, H, W, 1, device=device)
    v_r2l = torch.ones(B, H, W, 1, device=device)
    v_l2m = torch.ones(B, H, W, 1, device=device)
    v_r2m = torch.ones(B, H, W, 1, device=device)

    batch['left_pm'] = torch.cat([left_xyz, v_left], dim=-1)
    batch['right_pm'] = torch.cat([right_xyz, v_right], dim=-1)
    batch['motion_gt'] = {
        'l2r': torch.cat([l2r_xyz, v_l2r], dim=-1),
        'r2l': torch.cat([r2l_xyz, v_r2l], dim=-1),
        'l2m': torch.cat([l2m_xyz, v_l2m], dim=-1),
        'r2m': torch.cat([r2m_xyz, v_r2m], dim=-1),
    }

    # Ensure we include: index 0 as mid-time (not 0/1), plus 1.0 and 0.0
    batch['query_times'] = torch.tensor([[0.5, 1.0, 0.0]], device=device)

    return batch


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    """Main overfitting test using model.get_loss with synthetic GT."""
    print("Starting get_loss overfitting test...")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Override config for testing
    config.model.use_pretrained = False  # Don't load pretrained weights for this test
    config.model.head_type = 'linear'  # Use simpler linear head for faster overfitting
    config.model.img_size = [16, 16]  # Smaller images for faster testing
    config.model.enc_embed_dim = 64  # Smaller model for faster testing
    config.model.enc_depth = 4
    config.model.enc_num_heads = 4
    config.model.dec_embed_dim = 384 // 8
    config.model.dec_depth = 10
    config.model.dec_num_heads = 4
    # Disable confidence for this run (we will compute pure L2 loss in-test)
    # config.model.conf_mode = None
    # config.model.motion_conf_mode = None
    

    # Create model
    model = get_model(config, device)
    model = model.to(device)  # Ensure model is on correct device
    # Use eval() to keep features deterministic (no dropout) while still training head params
    # model.eval()
    model.train()
    
    # Create constant inputs and target
    print("Creating constant batch and synthetic GT...")
    batch = create_constant_batch(batch_size=1, config=config, device=device)  # Small batch for faster overfitting
    batch = create_synthetic_pointmaps_and_motion(batch, config=config, device=device)
    
    print(f"Input shapes:")
    print(f"  left_image: {batch['left_image'].shape}")
    print(f"  right_image: {batch['right_image'].shape}")
    print(f"  left_pm: {batch['left_pm'].shape}")
    print(f"  right_pm: {batch['right_pm'].shape}")
    print(f"  query_times: {batch['query_times'].shape} -> {batch['query_times'].tolist()}")
    
    # IMPORTANT: make params a list so the optimizer and grad clipping both see them
    params = [p for p in model.parameters() if p.requires_grad]
    
    # # Setup optimizer - unfreeze last 1-2 decoder blocks (left stream) and left head
    # for p in model.parameters():
    #     p.requires_grad = False
    # params = []
    # # left head - handle both linear and DPT heads
    # try:
    #     # Try linear head first
    #     for p in model.downstream_head1.proj.parameters():
    #         p.requires_grad = True
    #         params.append(p)
    # except AttributeError:
    #     # If no .proj attribute, get all head parameters (for DPT heads)
    #     for p in model.downstream_head1.parameters():
    #         p.requires_grad = True
    #         params.append(p)
    # # last one or two decoder blocks for left stream
    # unfreeze_blocks = []
    # try:
    #     unfreeze_blocks = [
    #         model.dec_blocks[-1], 
    #         model.dec_blocks[-2], 
    #         model.dec_blocks[-3],
    #         model.dec_blocks[-4],
    #         model.dec_blocks[-5],
    #         model.dec_blocks[-6],
    #         model.dec_blocks[-7],
    #         # model.dec_blocks[-8],
    #         # model.dec_blocks[-9],
    #         # model.dec_blocks[-10]
    #     ]
    # except Exception:
    #     unfreeze_blocks = [model.dec_blocks[-1]]
    # for blk in unfreeze_blocks:
    #     for p in blk.parameters():
    #         p.requires_grad = True
    #         params.append(p)

    # optimizer = get_optimizer(params, config)
    optimizer = torch.optim.Adam(params, lr=1e-5)
    
    # Training loop
    print("\nStarting overfitting loop...")
    num_iterations = 50000  # More iterations for exact fit
    print_every = 25     # Print less frequently
    target_loss = 1e-5
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        # Use the copied get_loss function which returns L2 when conf is disabled
        loss, loss_details = get_loss(model, batch, outputs)
        
        # Backward pass with gradient clipping
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        
        # Print progress
        if (iteration + 1) % print_every == 0 or iteration == 0:
            # Basic loss info
            print(f"{iteration + 1:4d}/{num_iterations}: total_loss = {loss.item():.10f} | grad norm = {float(grad_norm):.10f}")

            # Show all available metrics
            print("  All metrics:")
            for key, value in loss_details.items():
                print(f"    {key}: {value:.8f}")

            # Check if we've achieved target L2 mean
            if loss_details['left_l2_mean'] < target_loss:
                print(f"✅ Target L2 mean {target_loss} achieved at iteration {iteration + 1}!")
                break
    
    # Final evaluation
    print(f"\nFinal evaluation:")
    print(f"Final total loss: {loss.item():.10f}")

    # Show final metrics
    print("Final metrics:")
    for key, value in loss_details.items():
        print(f"  {key}: {value:.8f}")

    # Check exact matching
    with torch.no_grad():
        final_outputs = model(batch)
        final_loss, final_details = get_loss(model, batch, final_outputs)

        print(f"\nEvaluation total loss: {final_loss.item():.10f}")
        print("Evaluation metrics:")
        for key, value in final_details.items():
            print(f"  {key}: {value:.8f}")
        
        # Compute element-wise differences
        # Compare postprocessed left prediction vs left GT for a small patch
        left_pred_pp = reg_dense_depth(final_outputs['left_map_pred'], model.depth_post_mode)
        diff = torch.abs(left_pred_pp - batch['left_pm'][..., :3])
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Final loss: {final_loss.item():.10f}")
        print(f"Max absolute difference: {max_diff:.10f}")
        print(f"Mean absolute difference: {mean_diff:.10f}")
        
        # Success criteria
        success = final_loss.item() < target_loss
        
        if success:
            print("✅ SUCCESS: Model successfully overfitted using get_loss!")
        else:
            print("❌ FAILURE: Left head did not achieve target accuracy.")
            print(f"   Target loss: {target_loss}")
            print(f"   Achieved loss: {final_loss.item():.10f}")
        
        # Additional verification: check that prediction is very close to target
        print(f"\nSample comparison (first 3x3 patch, channel x):")
        print(f"Target   left_pm[0,0:3,0:3,0]: {batch['left_pm'][0, 0:3, 0:3, 0].detach().cpu().numpy()}")
        print(f"Predicted left_pp[0,0:3,0:3,0]: {left_pred_pp[0, 0:3, 0:3, 0].detach().cpu().numpy()}")
        
        return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
# coding: utf-8
"""
Author: Kevin Mathew T
Date: 2025-09-29
LinkedIn: https://www.linkedin.com/in/kevinmathewt/

Purpose:
- Run DynaDUSt3R on a folder of two-frame GIF "paper examples" (frame 0=left, frame 1=right)
- Save basic visualization outputs (depth/confidence) per GIF
- Stream a short time sequence to Rerun using utils.rerun_viz.visualize_sequence_from_pms

Strict assumptions version:
- All inputs, files, and model outputs exist and are valid.
- No optional branches, no fallbacks, no warnings.

usage: poetry run python -m infer.infer_example
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import hydra
from omegaconf import DictConfig

from models import get_model
from models.dust3r.utils.heads.postprocess import reg_dense_depth
from utils.rerun_viz import visualize_trajectories_from_pms
from utils.viz import make_depth_and_disp_imgs, conf_to_gray_img


def _select_device(cfg_device: str) -> torch.device:
    if cfg_device == "cuda" or cfg_device == "auto":
        # Assume CUDA is available
        return torch.device("cuda")
    # Otherwise it's explicitly "cpu"
    return torch.device("cpu")


def _load_gif_first_two_frames(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load first two frames from a GIF as RGB uint8 arrays (assumes at least 2 frames)."""
    with Image.open(path) as im:
        im.seek(0)
        left = np.array(im.convert("RGB"))
        im.seek(1)
        right = np.array(im.convert("RGB"))
    return left, right


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.array(Image.fromarray(img).resize((size, size), resample=Image.BILINEAR))


def _to_model_tensor(img_uint8_hw3: np.ndarray, *, normalize_m1p1: bool) -> torch.Tensor:
    """HWC uint8 [0,255] -> CHW float32 in either [0,1] or [-1,1]."""
    t = torch.from_numpy(img_uint8_hw3).to(dtype=torch.float32).permute(2, 0, 1) / 255.0
    if normalize_m1p1:
        t = (t - 0.5) / 0.5
    return t


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_basic_outputs(out_dir: Path,
                        base_name: str,
                        pred_left_hw3: np.ndarray,
                        pred_right_hw3: np.ndarray,
                        conf_left_hw1: np.ndarray,
                        conf_right_hw1: np.ndarray):
    """Save depth and confidence images for left/right predictions (assumes valid inputs)."""
    valid_all_left = np.ones(pred_left_hw3.shape[:2], dtype=bool)
    valid_all_right = np.ones(pred_right_hw3.shape[:2], dtype=bool)

    left_depth_rgb, _ = make_depth_and_disp_imgs(pred_left_hw3, valid_all_left)
    right_depth_rgb, _ = make_depth_and_disp_imgs(pred_right_hw3, valid_all_right)

    _ensure_dir(out_dir)
    Image.fromarray(left_depth_rgb).save(out_dir / f"{base_name}_left_depth_pred.png")
    Image.fromarray(right_depth_rgb).save(out_dir / f"{base_name}_right_depth_pred.png")

    conf_left_img = conf_to_gray_img(conf_left_hw1, valid_all_left)
    conf_right_img = conf_to_gray_img(conf_right_hw1, valid_all_right)
    Image.fromarray(conf_left_img).save(out_dir / f"{base_name}_left_conf.png")
    Image.fromarray(conf_right_img).save(out_dir / f"{base_name}_right_conf.png")


def _load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Strict load: assumes file exists and contains a matching 'state_dict'."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    # Assume training used [-1,1] normalization; set a flag for the caller
    setattr(model, "_normalize_m1p1", True)
    return model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    # Device
    device = _select_device(str(config.infer.device))

    # Model + checkpoint (assumes a valid checkpoint path)
    model = get_model(config, device).to(device).eval()
    ckpt_path = Path(str(config.infer.checkpoint_path))
    model = _load_checkpoint_into_model(model, ckpt_path, device)

    # IO
    input_dir = Path(str(config.infer.input_dir))
    output_dir = Path(str(config.infer.output_dir))
    _ensure_dir(output_dir)
    
    # Rerun visualization output directory
    rerun_dir = Path("/scratch/km6748/vision-experiments/scratch/rerun_visualizations")
    _ensure_dir(rerun_dir)

    gif_glob = str(config.infer.gif_glob)
    gif_paths = sorted(input_dir.glob(gif_glob))
    
    # If debug mode, only process first example
    if config.debug:
        gif_paths = [gif_paths[-1]]

    # Query times (assume >= 2)
    num_query_times = int(config.infer.num_query_times)
    query_times_for_model = torch.linspace(0.0, 1.0, steps=num_query_times, dtype=torch.float32, device=device)

    # Target input size
    target_size = int(config.data.size)

    for gif_path in gif_paths:
        # Load first two frames
        left, right = _load_gif_first_two_frames(gif_path)

        # Resize
        left_r = _resize_to_square(left, target_size)
        right_r = _resize_to_square(right, target_size)

        # Build batch (assume [-1,1] normalization as set above)
        normalize_m1p1 = bool(getattr(model, "_normalize_m1p1", True))
        left_tensor = _to_model_tensor(left_r, normalize_m1p1=normalize_m1p1).unsqueeze(0).to(device)
        right_tensor = _to_model_tensor(right_r, normalize_m1p1=normalize_m1p1).unsqueeze(0).to(device)

        batch = {
            "left_image": left_tensor,
            "right_image": right_tensor,
            "query_times": query_times_for_model,  # (T,)
            "left_instance": [f"{gif_path.stem}_L"],
            "right_instance": [f"{gif_path.stem}_R"],
        }

        # Forward
        with torch.inference_mode():
            outputs = model(batch)

        # Extract predictions (strict keys/shapes)
        left_pred = outputs["left_map_pred"][0].detach().cpu().numpy()                    # (H,W,3)
        right_pred = outputs["right_map_pred_in_left_frame"][0].detach().cpu().numpy()    # (H,W,3)

        c_left = outputs["left_map_pred_conf"][0].detach().cpu().numpy()
        c_right = outputs["right_map_pred_conf"][0].detach().cpu().numpy()
        conf_left_np = c_left[..., 0] if c_left.ndim == 3 and c_left.shape[-1] == 1 else c_left
        conf_right_np = c_right[..., 0] if c_right.ndim == 3 and c_right.shape[-1] == 1 else c_right

        # Confidence-based filtering for Rerun visualization
        # Check for per-file override
        gif_stem = gif_path.stem
        overrides = config.infer.get("conf_percentile_overrides", {})
        if gif_stem in overrides:
            conf_percentile = float(overrides[gif_stem])
        else:
            conf_percentile = float(config.infer.conf_percentile_threshold)
        threshold_value = np.percentile(conf_left_np, conf_percentile * 100.0)
        conf_mask = (conf_left_np >= threshold_value)

        # Post depth mapping (strict)
        left_pred_pp = reg_dense_depth(outputs["left_map_pred"][0], model.depth_post_mode).detach().cpu().numpy()
        right_pred_pp = reg_dense_depth(outputs["right_map_pred_in_left_frame"][0], model.depth_post_mode).detach().cpu().numpy()

        # Save outputs
        base = gif_path.stem
        out_dir = output_dir / base
        _save_basic_outputs(out_dir, base, left_pred_pp, right_pred_pp, conf_left_np, conf_right_np)

        # Sequence for Rerun (assumes full motion_pred present)
        motion_pred = outputs["motion_pred"]
        T = int(query_times_for_model.shape[0])

        disps = []
        for t_idx in range(T):
            if t_idx == 0:
                disp_np = np.zeros_like(left_pred_pp, dtype=left_pred_pp.dtype)
            elif t_idx == T - 1:
                disp_np = motion_pred["l_to_r"][0].detach().cpu().numpy()[..., :3]
            else:
                disp_np = motion_pred[f"l_to_t{t_idx}"][0].detach().cpu().numpy()[..., :3]
            disps.append(disp_np)

        # Build point maps with validity channel (4th channel = confidence mask)
        pms_seq = [np.concatenate([left_pred_pp + d, conf_mask[..., None].astype(np.float32)], axis=-1) 
                   for d in disps]
        
        # Build motion sequence with validity channel
        motion_deltas = [disps[i + 1] - disps[i] for i in range(T - 1)]
        motion_seq = np.stack([np.concatenate([md, conf_mask[..., None].astype(np.float32)], axis=-1) 
                               for md in motion_deltas], axis=0)
        image_seq = [left_r for _ in range(T)]

        # Save rerun visualization
        rerun_file = rerun_dir / f"{base}_sequence.rrd"
        # Top-percentile persistent trajectories
        traj_percentile = float(config.infer.get("traj_percentile_threshold", 0.75))
        visualize_trajectories_from_pms(pms_seq, motion_seq, image_seq=image_seq, 
                                        name=f"paper_seq_{base}", traj_percentile_threshold=traj_percentile) #, save=True, path=str(rerun_file))

        print(f"Processed {gif_path}")


if __name__ == "__main__":
    main()

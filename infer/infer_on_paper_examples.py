#!/usr/bin/env python
# coding: utf-8
"""
Inference script for DynaDUSt3R on paper examples (GIFs).
Each GIF contains two frames: first frame = left image, second frame = right image.

poetry run python -m infer.infer_on_paper_examples
"""

import os
import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from models import get_model
import utils.viz as viz_utils


def load_gif_frames(gif_path):
    """Load first two frames from GIF: first = left, second = right."""
    with Image.open(gif_path) as img:
        # First frame (left image)
        left_frame = np.array(img.convert('RGB'))

        # Second frame (right image)
        try:
            img.seek(1)
            right_frame = np.array(img.convert('RGB'))
        except EOFError:
            # If only one frame, duplicate it (fallback)
            print(f"Warning: {gif_path} has only one frame, duplicating for right image")
            right_frame = left_frame.copy()

    return left_frame, right_frame


def process_image_for_model(img_array, device):
    """
    Process image array to match model input format.
    Converts (H, W, 3) numpy array to (3, H, W) torch tensor with proper normalization.
    """
    # Convert to torch tensor and move to device
    img = torch.from_numpy(img_array).float().to(device)

    # Convert HWC to CHW if needed
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)

    # Normalize to [-1, 1] range (same as dataloader)
    img = img / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    img = (img - mean) / std

    return img


def create_inference_batch(left_frame, right_frame, device, gif_name):
    """
    Create batch dictionary matching the model's forward() expectations.
    Based on DynaDUSt3R.forward() method requirements.
    """
    # Process images
    left_image = process_image_for_model(left_frame, device)  # (3, H, W)
    right_image = process_image_for_model(right_frame, device)  # (3, H, W)

    # Add batch dimension
    left_image = left_image.unsqueeze(0)  # (1, 3, H, W)
    right_image = right_image.unsqueeze(0)  # (1, 3, H, W)

    # Create query times - for inference, we want motion predictions at various time points
    # Based on the dataloader, query_times should be [mid_relative, right_relative, left_relative]
    # For paper examples: mid=0.5 (middle time), right=1.0 (right image time), left=0.0 (left image time)
    query_times = torch.tensor([[0.5, 1.0, 0.0]], device=device)  # (1, 3)

    # Create dummy camera parameters (identity intrinsics/extrinsics)
    # For inference on arbitrary images, we use simple pinhole cameras
    H, W = left_frame.shape[:2]
    focal_length = W * 0.5  # Simple focal length assumption
    intrinsics = torch.tensor([[focal_length, 0, W/2],
                               [0, focal_length, H/2],
                               [0, 0, 1]], dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)  # Identity pose

    # Create batch dictionary
    batch = {
        "left_image": left_image,      # (B, 3, H, W)
        "right_image": right_image,    # (B, 3, H, W)
        "query_times": query_times,    # (B, T) - optional, but needed for motion prediction
        "left_instance": [f"{gif_name}_left"],
        "right_instance": [f"{gif_name}_right"],
        "cam": (intrinsics, extrinsics),          # Left camera (intrinsics, extrinsics)
        "cam_mid": (intrinsics, extrinsics),      # Mid camera (same as left for simplicity)
        "cam_right": (intrinsics, extrinsics),    # Right camera (same as left for simplicity)
    }

    return batch


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    # Get inference config
    infer_config = config.infer

    # Set device
    if infer_config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(infer_config.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from checkpoint: {infer_config.checkpoint_path}")
    model = get_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(infer_config.checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Create output directory
    output_dir = Path(infer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find all GIF files
    gif_pattern = os.path.join(infer_config.input_dir, infer_config.gif_glob)
    gif_files = glob.glob(gif_pattern)
    print(f"Found {len(gif_files)} GIF files: {gif_files}")

    # Debug mode: process only first GIF
    if config.debug:
        gif_files = gif_files[:1]
        print(f"Debug mode: processing only first GIF: {gif_files}")

    # Process each GIF
    with torch.no_grad():
        for gif_path in gif_files:
            gif_name = Path(gif_path).stem
            print(f"\nProcessing {gif_name}...")

            # Load frames from GIF
            left_frame, right_frame = load_gif_frames(gif_path)
            print(f"  Loaded frames: left {left_frame.shape}, right {right_frame.shape}")

            # Create batch for model
            batch = create_inference_batch(left_frame, right_frame, device, gif_name)

            # Forward pass
            print("  Running model forward pass...")
            outputs = model(batch)
            print(f"  Motion prediction keys: {list(outputs.get('motion_pred', {}).keys())}")

            # Create a minimal batch structure for visualization
            # The viz function expects certain fields, so we need to mock some of them
            viz_batch = {
                "left_image": batch["left_image"],    # (B, 3, H, W)
                "right_image": batch["right_image"],  # (B, 3, H, W)
                "mid_image": batch["right_image"],    # Mock mid image as right (for viz compatibility)
                "cam": batch["cam"],                  # Left camera
                "cam_mid": batch["cam_mid"],          # Mid camera
                "cam_right": batch["cam_right"],      # Right camera
            }

            # For visualization, we need to create fake point maps since we don't have ground truth
            # We'll create dummy point maps with some valid points for visualization
            H, W = left_frame.shape[:2]
            dummy_pm = torch.zeros(1, H, W, 4, device=device)
            dummy_pm[..., 3] = 1.0  # Set validity to 1

            viz_batch.update({
                "left_pm": dummy_pm.clone(),
                "mid_pm": dummy_pm.clone(),
                "right_pm": dummy_pm.clone(),
                "motion_gt": {
                    "l2m": dummy_pm.clone(),
                    "r2m": dummy_pm.clone(),
                    "l2r": dummy_pm.clone(),
                    "r2l": dummy_pm.clone(),
                }
            })

            # Call visualization function
            print("  Generating visualizations...")
            base_name = f"{gif_name}_inference"

            # For now, disable motion-related visualizations due to camera parameter issues
            # Save only basic depth/conf visualizations
            try:
                viz_utils.save_visualizations(
                    viz_batch,
                    outputs,
                    base_name,
                    i=0,  # First (and only) batch item
                    depth_post_mode=getattr(model, 'depth_post_mode', None),
                    motion_depth_post_mode=getattr(model, 'motion_depth_post_mode', None),
                    save_dir=str(output_dir)
                )
            except Exception as viz_error:
                print(f"  Warning: Visualization failed: {viz_error}")
                print("  Continuing with next GIF...")

            print(f"  Saved visualizations for {gif_name}")

    print(f"\nInference complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

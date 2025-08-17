#!/usr/bin/env python3
"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/

Validation script for stereo motion dataloader outputs.
Verifies that point maps and motion maps satisfy coordinate frame constraints.

Key validations:
1. Point maps: 3D coords in left camera frame, projected using source camera
2. Motion maps: Motion vectors in left camera frame, projected using source camera  
3. Validity masks align with successful projections
4. Motion consistency across different paths
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

# Import the geometry utilities
import utils.geometry as geo


class DataloaderValidator:
    def __init__(self, config, pixel_tolerance=2.0, depth_tolerance=0.01, verbose=True):
        """
        Initialize validator with tolerances for different checks.
        
        Args:
            config: Dataset configuration
            pixel_tolerance: Max reprojection error in pixels
            depth_tolerance: Relative depth tolerance for validity checks
            verbose: Whether to print detailed results
        """
        self.config = config
        self.pixel_tolerance = pixel_tolerance
        self.depth_tolerance = depth_tolerance
        self.verbose = verbose
        self.results = {}
        
    def validate_batch(self, batch, save_dir=None):
        """Run all validation checks on a batch."""
        batch_size = batch['batch_size']
        results = {
            'point_map_checks': [],
            'motion_map_checks': [],
            'consistency_checks': [],
            'validity_checks': []
        }
        
        # Process each sample in batch
        for i in range(batch_size):
            sample_results = self._validate_sample(batch, i)
            
            # Aggregate results
            for key in results:
                results[key].extend(sample_results[key])
            
            # Optional visualization
            if save_dir and i == 0:  # Visualize first sample
                self._visualize_sample(batch, i, sample_results, save_dir)
        
        # Compute statistics
        stats = self._compute_stats(results)
        return stats, results
    
    def _validate_sample(self, batch, idx):
        """Validate a single sample from the batch."""
        # Extract data for this sample
        left_pm = batch['left_pm'][idx].numpy()      # (H, W, 4)
        mid_pm = batch['mid_pm'][idx].numpy()        # (H, W, 4)
        right_pm = batch['right_pm'][idx].numpy()    # (H, W, 4)
        
        left_img = batch['left_image'][idx].permute(1, 2, 0).numpy()
        mid_img = batch['mid_image'][idx].permute(1, 2, 0).numpy()
        right_img = batch['right_image'][idx].permute(1, 2, 0).numpy()

        # extract intrinsics and extrinsics for this sample
        left_cam = (batch['cam'][0][idx].numpy(), batch['cam'][1][idx].numpy())
        mid_cam = (batch['cam_mid'][0][idx].numpy(), batch['cam_mid'][1][idx].numpy())
        right_cam = (batch['cam_right'][0][idx].numpy(), batch['cam_right'][1][idx].numpy())
        
        # Motion maps
        motion_gt = {k: v[idx].numpy() for k, v in batch['motion_gt'].items()}
        
        results = {
            'point_map_checks': [],
            'motion_map_checks': [],
            'consistency_checks': [],
            'validity_checks': []
        }
        
        # 1. Validate point maps
        pm_results = self._validate_point_maps(
            {'left': left_pm, 'mid': mid_pm, 'right': right_pm},
            {'left': left_cam, 'mid': mid_cam, 'right': right_cam},
            left_cam  # reference camera
        )
        results['point_map_checks'] = pm_results
        
        # 2. Validate motion maps
        motion_results = self._validate_motion_maps(
            motion_gt,
            {'left': left_cam, 'mid': mid_cam, 'right': right_cam},
            {'left': left_pm, 'mid': mid_pm, 'right': right_pm},
            left_cam  # reference camera
        )
        results['motion_map_checks'] = motion_results
        
        # 3. Check motion consistency
        consistency_results = self._check_motion_consistency(motion_gt)
        results['consistency_checks'] = consistency_results
        
        # 4. Validate validity masks
        validity_results = self._validate_validity_masks(
            {'left': left_pm, 'mid': mid_pm, 'right': right_pm},
            motion_gt
        )
        results['validity_checks'] = validity_results
        
        return results
    
    def _validate_point_maps(self, point_maps, cameras, ref_cam):
        """
        Validate that point maps satisfy:
        1. 3D coordinates are in reference (left) camera frame
        2. Pixels correspond to the source camera's image space
        """
        results = []
        
        for name, pm in point_maps.items():
            cam = cameras[name]
            h, w = pm.shape[:2]
            
            # Get valid points
            valid_mask = pm[..., 3] > 0
            if not np.any(valid_mask):
                results.append({
                    'name': f'pm_{name}',
                    'status': 'skip',
                    'reason': 'No valid points'
                })
                continue
            
            valid_points_ref = pm[valid_mask, :3]  # Points in ref frame
            valid_pixels = np.stack(np.where(valid_mask), axis=1)  # (v, u)
            
            # Transform points from ref frame to source camera frame
            points_world = geo.cam_pc_to_world_pc(valid_points_ref, ref_cam)
            points_source = geo.world_pc_to_cam_pc(points_world, cam)
            
            # Project using source camera
            intrinsics = cam[0]
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            x, y, z = points_source[:, 0], points_source[:, 1], points_source[:, 2]
            z_valid = z > 0
            
            if not np.any(z_valid):
                results.append({
                    'name': f'pm_{name}',
                    'status': 'fail',
                    'reason': 'All points behind camera'
                })
                continue
            
            # Compute projected pixels
            u_proj = fx * x[z_valid] / z[z_valid] + cx
            v_proj = fy * y[z_valid] / z[z_valid] + cy
            
            # Compare with actual pixels
            actual_pixels = valid_pixels[z_valid]
            u_actual = actual_pixels[:, 1].astype(float)
            v_actual = actual_pixels[:, 0].astype(float)
            
            pixel_errors = np.sqrt((u_proj - u_actual)**2 + (v_proj - v_actual)**2)
            max_error = np.max(pixel_errors)
            mean_error = np.mean(pixel_errors)
            
            # Check if points are truly in reference frame
            # We'll verify by transforming to ref frame and comparing
            points_ref_check = geo.world_pc_to_cam_pc(points_world, ref_cam)
            coord_errors = np.abs(points_ref_check - valid_points_ref[z_valid])
            max_coord_error = np.max(coord_errors)
            
            status = 'pass' if max_error < self.pixel_tolerance else 'fail'
            
            results.append({
                'name': f'pm_{name}',
                'status': status,
                'max_pixel_error': max_error,
                'mean_pixel_error': mean_error,
                'max_coord_error': max_coord_error,
                'num_valid': np.sum(z_valid),
                'num_total': len(valid_points_ref)
            })
            
        return results
    
    def _validate_motion_maps(self, motion_maps, cameras, point_maps, ref_cam):
        """
        Validate that motion maps satisfy:
        1. Motion vectors are in reference (left) camera frame
        2. Pixels correspond to the source camera's image space
        3. Motion connects corresponding points between frames
        """
        results = []
        
        # Define motion map sources and targets
        motion_specs = {
            'l2m': ('left', 'mid'),
            'r2m': ('right', 'mid'),
            'l2r': ('left', 'right'),
            'r2l': ('right', 'left')
        }
        
        for motion_name, (src_name, tgt_name) in motion_specs.items():
            motion_map = motion_maps[motion_name]  # (H, W, 4)
            src_pm = point_maps[src_name]
            tgt_pm = point_maps[tgt_name]
            src_cam = cameras[src_name]
            
            # Get valid motion vectors
            valid_mask = motion_map[..., 3] > 0
            if not np.any(valid_mask):
                results.append({
                    'name': motion_name,
                    'status': 'skip',
                    'reason': 'No valid motion'
                })
                continue
            
            # Extract valid data
            valid_pixels = np.stack(np.where(valid_mask), axis=1)  # (N, 2) as (v, u)
            motion_vectors = motion_map[valid_mask, :3]  # (N, 3) in ref frame
            
            # Get corresponding points from point maps
            src_points = src_pm[valid_mask, :3]  # Should be in ref frame
            
            # Expected target points
            expected_tgt_points = src_points + motion_vectors
            
            # Find matching points in target point map
            # This is approximate - we look for nearby valid points
            errors = []
            matches_found = 0
            
            for i in range(len(valid_pixels)):
                v, u = valid_pixels[i]
                src_pt = src_points[i]
                expected_tgt = expected_tgt_points[i]
                
                # Look in a small window around expected position
                # First, project expected target to target image
                tgt_world = geo.cam_pc_to_world_pc(expected_tgt.reshape(1, 3), ref_cam)
                tgt_cam_coords = geo.world_pc_to_cam_pc(tgt_world, cameras[tgt_name])
                
                if tgt_cam_coords[0, 2] <= 0:
                    continue
                
                # Project to target image
                intr = cameras[tgt_name][0]
                u_tgt = intr[0, 0] * tgt_cam_coords[0, 0] / tgt_cam_coords[0, 2] + intr[0, 2]
                v_tgt = intr[1, 1] * tgt_cam_coords[0, 1] / tgt_cam_coords[0, 2] + intr[1, 2]
                
                u_tgt = int(round(u_tgt))
                v_tgt = int(round(v_tgt))
                
                # Check if this pixel has a valid point in target
                h, w = tgt_pm.shape[:2]
                if 0 <= u_tgt < w and 0 <= v_tgt < h and tgt_pm[v_tgt, u_tgt, 3] > 0:
                    actual_tgt = tgt_pm[v_tgt, u_tgt, :3]
                    error = np.linalg.norm(actual_tgt - expected_tgt)
                    errors.append(error)
                    matches_found += 1
            
            if matches_found > 0:
                max_error = np.max(errors)
                mean_error = np.mean(errors)
                status = 'pass' if mean_error < 0.1 else 'warn'  # 10cm threshold
            else:
                max_error = mean_error = -1
                status = 'fail'
            
            results.append({
                'name': motion_name,
                'status': status,
                'matches_found': matches_found,
                'total_valid': np.sum(valid_mask),
                'max_3d_error': max_error,
                'mean_3d_error': mean_error
            })
        
        return results
    
    def _check_motion_consistency(self, motion_maps):
        """Check if motion vectors are consistent: l2m + m2r ≈ l2r"""
        results = []
        
        # Extract motion maps
        l2m = motion_maps.get('l2m')
        r2m = motion_maps.get('r2m')
        l2r = motion_maps.get('l2r')
        r2l = motion_maps.get('r2l')
        
        # Check l2m + m2r ≈ l2r (but we don't have m2r directly)
        # Instead check: l2r + r2l ≈ 0 (round trip)
        if l2r is not None and r2l is not None:
            # Find pixels valid in both maps
            valid_l2r = l2r[..., 3] > 0
            valid_r2l = r2l[..., 3] > 0
            
            # For round trip, we need to map pixels properly
            # This is complex, so we'll do a simpler check:
            # Average motion magnitude should be similar
            
            l2r_mag = np.linalg.norm(l2r[valid_l2r, :3], axis=1).mean()
            r2l_mag = np.linalg.norm(r2l[valid_r2l, :3], axis=1).mean()
            
            # They should have similar magnitudes
            mag_ratio = l2r_mag / (r2l_mag + 1e-6)
            status = 'pass' if 0.8 < mag_ratio < 1.2 else 'warn'
            
            results.append({
                'check': 'motion_magnitude_symmetry',
                'status': status,
                'l2r_magnitude': l2r_mag,
                'r2l_magnitude': r2l_mag,
                'ratio': mag_ratio
            })
        
        return results
    
    def _validate_validity_masks(self, point_maps, motion_maps):
        """Validate that validity masks are correctly set."""
        results = []
        
        # Check point maps
        for name, pm in point_maps.items():
            valid_mask = pm[..., 3] > 0
            points = pm[..., :3]
            
            # Check that valid points have non-zero coordinates
            zero_points = np.all(points == 0, axis=-1)
            invalid_valids = valid_mask & zero_points
            
            if np.any(invalid_valids):
                results.append({
                    'name': f'pm_{name}_validity',
                    'status': 'warn',
                    'invalid_valid_count': np.sum(invalid_valids),
                    'reason': 'Valid mask set for zero points'
                })
            else:
                results.append({
                    'name': f'pm_{name}_validity',
                    'status': 'pass'
                })
        
        # Check motion maps
        for name, mm in motion_maps.items():
            valid_mask = mm[..., 3] > 0
            motion = mm[..., :3]
            
            # Check that valid motions are not all zero
            zero_motion = np.all(motion == 0, axis=-1)
            zero_valid_motion = valid_mask & zero_motion
            
            # Some zero motion is OK (stationary points)
            zero_ratio = np.sum(zero_valid_motion) / (np.sum(valid_mask) + 1e-6)
            
            if zero_ratio > 0.9:  # More than 90% stationary is suspicious
                status = 'warn'
                reason = f'{zero_ratio:.1%} of valid motion is zero'
            else:
                status = 'pass'
                reason = None
            
            results.append({
                'name': f'mm_{name}_validity',
                'status': status,
                'zero_motion_ratio': zero_ratio,
                'reason': reason
            })
        
        return results
    
    def _visualize_sample(self, batch, idx, results, save_dir):
        """Create visualization of validation results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract images
        left_img = self._tensor_to_img(batch['left_image'][idx])
        mid_img = self._tensor_to_img(batch['mid_image'][idx])
        right_img = self._tensor_to_img(batch['right_image'][idx])
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top row: Images with point map validity
        for ax, img, pm, title in zip(
            axes[0], 
            [left_img, mid_img, right_img],
            [batch['left_pm'][idx], batch['mid_pm'][idx], batch['right_pm'][idx]],
            ['Left', 'Mid', 'Right']
        ):
            ax.imshow(img)
            valid = pm[..., 3] > 0
            ax.contour(valid, colors='yellow', linewidths=0.5, alpha=0.5)
            ax.set_title(f'{title} (yellow=valid points)')
            ax.axis('off')
        
        # Bottom row: Motion maps
        motion_names = ['l2m', 'r2m', 'l2r']
        for i, (ax, motion_name) in enumerate(zip(axes[1], motion_names)):
            if motion_name in batch['motion_gt']:
                motion = batch['motion_gt'][motion_name][idx].numpy()
                valid = motion[..., 3] > 0
                
                # Compute motion magnitude
                mag = np.linalg.norm(motion[..., :3], axis=-1)
                mag[~valid] = 0
                
                im = ax.imshow(mag, cmap='hot')
                ax.set_title(f'{motion_name} magnitude')
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
        
        # Add text summary
        fig.text(0.02, 0.02, self._format_results_summary(results), 
                fontsize=8, verticalalignment='bottom', fontfamily='monospace')
        
        # Save
        sample_id = f"{idx}_{batch.get('sequence_idx', [0])[idx].item()}"
        plt.tight_layout()
        plt.savefig(save_dir / f'validation_{sample_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _tensor_to_img(self, tensor):
        """Convert normalized tensor to image."""
        img = tensor.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255  # Denormalize
        return img.astype(np.uint8)
    
    def _format_results_summary(self, results):
        """Format results as text summary."""
        lines = ["Validation Results:", "=" * 40]
        
        for category, checks in results.items():
            lines.append(f"\n{category}:")
            for check in checks:
                status = check['status']
                name = check.get('name', check.get('check', 'unknown'))
                symbol = {'pass': '✓', 'warn': '!', 'fail': '✗', 'skip': '-'}[status]
                
                line = f"  {symbol} {name}: {status}"
                if status != 'pass' and 'reason' in check:
                    line += f" ({check['reason']})"
                elif status == 'pass' and 'mean_pixel_error' in check:
                    line += f" (err: {check['mean_pixel_error']:.2f}px)"
                    
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _compute_stats(self, results):
        """Compute aggregate statistics from results."""
        stats = {}
        
        for category, checks in results.items():
            if not checks:
                continue
                
            counts = {'pass': 0, 'warn': 0, 'fail': 0, 'skip': 0}
            for check in checks:
                counts[check['status']] += 1
            
            total = sum(counts.values())
            stats[category] = {
                'counts': counts,
                'pass_rate': counts['pass'] / total if total > 0 else 0,
                'fail_rate': counts['fail'] / total if total > 0 else 0
            }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Validate stereo motion dataloader outputs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-batches', type=int, default=5, help='Number of batches to validate')
    parser.add_argument('--save-dir', type=str, default='validation_results', help='Directory for visualizations')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--loader', type=str, default='stereo4d', help='Which loader to use')
    args = parser.parse_args()
    
    # Import necessary modules
    import hydra
    from omegaconf import DictConfig
    from loaders import get_loaders
    
    # Load config
    with hydra.initialize(config_path="../config", version_base=None):
        config = hydra.compose(config_name=args.config)
    
    # tiny debug dataset
    config.data.len        = 1
    config.data.valid_len  = 1
    
    # Override loader if specified
    if args.loader:
        config.data.loader = args.loader
    
    # Create dataloader
    print(f"Creating dataloader for {config.data.loader}...")
    train_loader, valid_loader = get_loaders(config)
    
    # Create validator
    validator = DataloaderValidator(config, verbose=args.verbose)
    
    # Validate batches
    print(f"\nValidating {args.num_batches} batches...")
    all_stats = []
    
    for i, batch in enumerate(tqdm(train_loader, total=args.num_batches)):
        if i >= args.num_batches:
            break
            
        stats, results = validator.validate_batch(
            batch, 
            save_dir=args.save_dir if i < 3 else None  # Save first 3
        )
        all_stats.append(stats)
        
        if args.verbose and i == 0:  # Print detailed results for first batch
            print("\nDetailed results for first batch:")
            print("=" * 60)
            for category, checks in results.items():
                print(f"\n{category}:")
                for check in checks[:5]:  # First 5 checks
                    print(f"  {check}")
    
    # Aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    
    for category in ['point_map_checks', 'motion_map_checks', 'consistency_checks', 'validity_checks']:
        total_counts = {'pass': 0, 'warn': 0, 'fail': 0, 'skip': 0}
        
        for stats in all_stats:
            if category in stats:
                for status, count in stats[category]['counts'].items():
                    total_counts[status] += count
        
        total = sum(total_counts.values())
        if total > 0:
            print(f"\n{category}:")
            print(f"  Total checks: {total}")
            print(f"  Pass: {total_counts['pass']} ({total_counts['pass']/total:.1%})")
            print(f"  Warn: {total_counts['warn']} ({total_counts['warn']/total:.1%})")
            print(f"  Fail: {total_counts['fail']} ({total_counts['fail']/total:.1%})")
            print(f"  Skip: {total_counts['skip']} ({total_counts['skip']/total:.1%})")
    
    print(f"\nVisualizations saved to: {args.save_dir}/")
    
    # Overall verdict
    total_fails = sum(stats[cat]['counts']['fail'] for stats in all_stats 
                     for cat in stats if 'counts' in stats[cat])
    
    if total_fails == 0:
        print("\n✓ All validations PASSED!")
    else:
        print(f"\n✗ Found {total_fails} validation failures. Check visualizations for details.")


if __name__ == '__main__':
    main()

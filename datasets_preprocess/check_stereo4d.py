#!/usr/bin/env python3
"""
check_stereo4d_files.py
-----------------------
Check if stereo4d dataset files are present and loadable.
Only report problematic sequences - missing or failing to load.
"""

from __future__ import annotations
import os
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from joblib import Parallel, delayed
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def check_sequence_availability(seq: str, train_dirs: tuple, test_dirs: tuple) -> dict:
    """
    Check if sequence exists and is loadable in train or test split.
    Only return issues - missing sequences or loading errors.
    
    Returns:
        dict with issues, or None if everything is fine
    """
    train_lefteye_dir, train_ann_dir = train_dirs
    test_lefteye_dir, test_ann_dir = test_dirs
    
    # Check where sequence exists
    train_video = train_lefteye_dir / f"{seq}-left_rectified.mp4"
    train_ann = train_ann_dir / f"{seq}.npz"
    test_video = test_lefteye_dir / f"{seq}-left_rectified.mp4"
    test_ann = test_ann_dir / f"{seq}.npz"
    
    train_has_both = train_video.exists() and train_ann.exists()
    test_has_both = test_video.exists() and test_ann.exists()
    
    # If sequence doesn't exist in either split
    if not train_has_both and not test_has_both:
        missing_info = []
        if not train_video.exists() and not test_video.exists():
            missing_info.append("video missing from both splits")
        if not train_ann.exists() and not test_ann.exists():
            missing_info.append("annotation missing from both splits")
        
        return {
            'seq': seq,
            'issue': 'missing_files',
            'details': '; '.join(missing_info)
        }
    
    # Check loading - try the split where both files exist
    def try_loading(video_path, ann_path, split_name):
        issues = []
        
        # Try video loading
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            _ = vr[0]  # Try to read first frame
        except Exception as e:
            issues.append(f"Video loading error in {split_name}: {str(e)}")
            issues.append(f"Video traceback: {traceback.format_exc()}")
        
        # Try annotation loading
        try:
            data = np.load(ann_path, allow_pickle=True)
            required_keys = ['track_lengths', 'track_indices', 'track_coordinates', 'camera2world']
            missing_keys = [key for key in required_keys if key not in data.files]
            if missing_keys:
                issues.append(f"Annotation missing keys in {split_name}: {missing_keys}")
        except Exception as e:
            issues.append(f"Annotation loading error in {split_name}: {str(e)}")
            issues.append(f"Annotation traceback: {traceback.format_exc()}")
        
        return issues
    
    all_issues = []
    
    # Check train split if files exist there
    if train_has_both:
        train_issues = try_loading(train_video, train_ann, "train")
        all_issues.extend(train_issues)
    
    # Check test split if files exist there
    if test_has_both:
        test_issues = try_loading(test_video, test_ann, "test")
        all_issues.extend(test_issues)
    
    # Only return if there are actual loading issues
    if all_issues:
        return {
            'seq': seq,
            'issue': 'loading_error',
            'details': '\n'.join(all_issues)
        }
    
    # Everything is fine, return None
    return None


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main function to check dataset files and report only problematic sequences."""
    
    # Load and filter CSV data
    meta_csv = Path(cfg.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
    stats_csv = Path(cfg.dataset.stereo4d.meta_dir) / "stats.csv"
    
    print("Loading CSV files...")
    meta_df = pd.read_csv(
        meta_csv,
        header=0,
        names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
    )
    stats_df = pd.read_csv(stats_csv, skipinitialspace=True).query(
        "displacement_percentage_50 > 0.10 and d_frame > 5*16"
    )
    
    print(f"Filtered to {len(stats_df)} sequences based on displacement and frame criteria")
    
    # Generate sequences from CSV
    sequences = []
    for _, row in stats_df.iterrows():
        vid, cid = row["ytid"], row["clipid"]
        ts = meta_df.loc[(meta_df.vid == vid) & (meta_df.clipid == cid), "timestamp"]
        if not ts.empty:
            seq = f"{vid}_{int(ts.values[0])}"
            sequences.append(seq)
    
    print(f"Generated {len(sequences)} total sequences from CSV data")
    
    # Setup directory paths
    train_lefteye_dir = Path(cfg.dataset.stereo4d.lefteye_dir) / cfg.dataset.stereo4d.train_split
    train_ann_dir = Path(cfg.dataset.stereo4d.path) / cfg.dataset.stereo4d.train_split
    test_lefteye_dir = Path(cfg.dataset.stereo4d.lefteye_dir) / cfg.dataset.stereo4d.valid_split
    test_ann_dir = Path(cfg.dataset.stereo4d.path) / cfg.dataset.stereo4d.valid_split
    
    print(f"\nChecking directories:")
    print(f"Train video: {train_lefteye_dir} (exists: {train_lefteye_dir.exists()})")
    print(f"Train ann: {train_ann_dir} (exists: {train_ann_dir.exists()})")
    print(f"Test video: {test_lefteye_dir} (exists: {test_lefteye_dir.exists()})")
    print(f"Test ann: {test_ann_dir} (exists: {test_ann_dir.exists()})")
    
    # Parallel checking - only get problematic sequences
    print(f"\nChecking {len(sequences)} sequences for issues...")
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(check_sequence_availability)(
            seq, 
            (train_lefteye_dir, train_ann_dir),
            (test_lefteye_dir, test_ann_dir)
        ) 
        for seq in tqdm(sequences, desc="Checking sequences")
    )
    
    # Filter out None results (sequences that are fine)
    problems = [r for r in results if r is not None]
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(problems)} problematic sequences out of {len(sequences)} total")
    print(f"Success rate: {100 * (len(sequences) - len(problems)) / len(sequences):.1f}%")
    print(f"{'='*80}")
    
    if not problems:
        print("🎉 All sequences are present and loadable!")
        return
    
    # Group problems by type
    missing_files = [p for p in problems if p['issue'] == 'missing_files']
    loading_errors = [p for p in problems if p['issue'] == 'loading_error']
    
    if missing_files:
        print(f"\n📁 MISSING FILES ({len(missing_files)} sequences):")
        print("-" * 50)
        for p in missing_files:
            print(f"❌ {p['seq']}: {p['details']}")
    
    if loading_errors:
        print(f"\n🚨 LOADING ERRORS ({len(loading_errors)} sequences):")
        print("-" * 50)
        for p in loading_errors:
            print(f"❌ {p['seq']}:")
            print(f"   {p['details']}")
            print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
check_stereo4d.py
-----------------
Simple check script to verify all Stereo4D sequences can be loaded properly
from WebDataset shards.
"""
# /scratch/projects/fouheylab/km6748/stereo4d-data/vids/train/0j5gAUsrj68_103436770-left_rectified.mp4
import os
import json
import io
import logging
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from utils.train_utils import create_symlink_for_wids_cache, seed_everything

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm
import wids
import hydra
from omegaconf import DictConfig

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)

# Constants (same as preprocessing script)
SPLIT = "train"
NUM_WORKERS = 32
TEST_N = None  # Set to a number to test only N sequences

def discover_sequences(cfg: DictConfig) -> list[str]:
    """Same sequence discovery logic as preprocessing script."""
    meta_csv = os.path.join(cfg.dataset.stereo4d.meta_dir,
                            "stereo4d_id_to_time_and_fov_metadata.csv")
    stats_csv = os.path.join(cfg.dataset.stereo4d.meta_dir, "stats.csv")

    meta = pd.read_csv(meta_csv, header=0,
                       names=["vid","clipid","timestamp",
                              "start_yaw","end_yaw","start_tilt","end_tilt"])
    stats = pd.read_csv(stats_csv, skipinitialspace=True)
    stats = stats.query("displacement_percentage_50 > 0.10 and d_frame > 5*16")

    def _has_mp4(row):
        vid, cid = row["ytid"], row["clipid"]
        ts = meta.loc[(meta.vid == vid) & (meta.clipid == cid), "timestamp"]
        if ts.empty: 
            return False
        seq = f"{vid}_{int(ts.values[0])}"
        return (Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4").exists()

    exist_mask = pd.Series([_has_mp4(row) for _, row in stats.iterrows()],
                           index=stats.index)
    stats = stats[exist_mask]

    seqs = []
    for _, r in stats.iterrows():
        ts = meta.loc[(meta.vid == r["ytid"]) & (meta.clipid == r["clipid"]), "timestamp"]
        if not ts.empty:
            seqs.append(f"{r['ytid']}_{int(ts.values[0])}")
    
    return sorted(set(seqs))

def check_sequence(seq: str, idx_ds, key_to_idx: dict) -> tuple[str, bool, str]:
    """Check if a single sequence can be loaded properly."""
    try:
        # Check if sequence exists in index
        if seq not in key_to_idx:
            return seq, False, "sequence not found in key_to_idx"
        
        # Load sample from WebDataset
        sample = idx_ds[key_to_idx[seq]]
        
        # Check video data
        video_data = sample[".video.mp4"]
        if not video_data:
            return seq, False, "empty video data"
        
        # Try to create video reader
        if hasattr(video_data, 'read'):
            video_file_obj = video_data
        else:
            video_file_obj = io.BytesIO(video_data)
        
        vr = VideoReader(video_file_obj, ctx=cpu(0))
        if len(vr) == 0:
            return seq, False, "video has 0 frames"
        
        # Check annotation data
        ann_data = sample[".ann.npz"]
        if not ann_data:
            return seq, False, "empty annotation data"
        
        # Try to load npz
        if hasattr(ann_data, 'read'):
            ann_bytes = ann_data.read()
        else:
            ann_bytes = ann_data
        
        ann = np.load(io.BytesIO(ann_bytes), allow_pickle=True)
        required_keys = ['track_lengths', 'track_indices', 'track_coordinates', 'camera2world']
        for key in required_keys:
            if key not in ann:
                return seq, False, f"missing annotation key: {key}"
        
        # Check intrinsics data
        intr_data = sample[".intr.npy"]
        if not intr_data:
            return seq, False, "empty intrinsics data"
        
        # Try to load intrinsics
        if hasattr(intr_data, 'read'):
            intr_bytes = intr_data.read()
        else:
            intr_bytes = intr_data
        
        intrinsics = np.load(io.BytesIO(intr_bytes), allow_pickle=True)
        if intrinsics.shape != (3, 3):
            return seq, False, f"invalid intrinsics shape: {intrinsics.shape}"
        
        # Clean up
        del vr, ann, intrinsics
        
        return seq, True, "ok"
        
    except Exception as e:
        return seq, False, f"exception: {str(e)}"

def check_sequence_worker(args):
    """Worker function for multiprocessing."""
    seq, idx_ds, key_to_idx = args
    return check_sequence(seq, idx_ds, key_to_idx)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    create_symlink_for_wids_cache()
    seed_everything(cfg.seed)

    print(f"SPLIT={SPLIT}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"TEST_N={TEST_N}")

    # Load WebDataset index
    wds_dir = Path(cfg.dataset.stereo4d.path) / "wds" / SPLIT
    idx_json = wds_dir / "stereo4d-idx.json"
    map_json = wds_dir / "key_to_idx.json"
    
    if not idx_json.exists():
        print(f"Error: Index file not found: {idx_json}")
        return
    
    if not map_json.exists():
        print(f"Error: Key mapping file not found: {map_json}")
        return
    
    print(f"Loading WebDataset index from {idx_json}...")
    idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
    print(f"Loaded {len(idx_ds)} samples")
    
    with open(map_json) as f:
        key_to_idx = json.load(f)
    print(f"Loaded key_to_idx mapping with {len(key_to_idx)} entries")
    
    # Discover sequences
    seqs = discover_sequences(cfg)
    if TEST_N is not None:
        import random
        seqs = random.sample(seqs, min(TEST_N, len(seqs)))
    
    print(f"Checking {len(seqs)} sequences...")
    
    # Prepare args for multiprocessing
    args_list = [(seq, idx_ds, key_to_idx) for seq in seqs]
    
    # Check sequences in parallel
    failed_seqs = []
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(check_sequence_worker, args_list),
            total=len(args_list),
            desc="Checking sequences"
        ))
    
    # Process results
    total_checked = len(results)
    total_passed = 0
    
    for seq, success, message in results:
        if success:
            total_passed += 1
        else:
            failed_seqs.append((seq, message))
            print(f"FAILED: {seq} - {message}")
    
    print(f"\nResults:")
    print(f"Total sequences checked: {total_checked}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {len(failed_seqs)}")
    
    if failed_seqs:
        print(f"\nFailed sequences:")
        for seq, message in failed_seqs:
            print(f"  {seq}: {message}")
    else:
        print("All sequences passed!")

if __name__ == "__main__":
    main()
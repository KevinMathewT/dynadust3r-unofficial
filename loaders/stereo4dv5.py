import os, json, io, time, tempfile
from pathlib import Path
import weakref

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import webdataset as wds, wids
from decord import VideoReader, cpu

from .stereo_motion_base import StereoMotionBase
from .utils import geometry as geom


class Stereo4Dv5(StereoMotionBase):
    """
    Ultra-lightweight Stereo-4D dataloader optimized for 8TB+ datasets.
    
    NO CACHING - optimized for single-access patterns where each frame 
    is accessed exactly once. Minimizes memory usage and maximizes throughput.
    
    Key optimizations:
    - Direct frame access without loading full video
    - Single-frame annotation processing  
    - Minimal memory footprint
    - Vectorized operations where possible
    """

    def __init__(self, config, valid: bool = False):
        super().__init__(config, valid)  # Pass valid parameter to parent
        print("loading cacheless stereo4d v4 dataset (optimized for 8TB+ data)...")

        self.dataset_label = config.dataset.stereo4d.name
        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        self.config = config

        # WebDataset setup (same as original)
        wds_dir = Path(config.dataset.stereo4d.path) / "wds" / split
        idx_json = wds_dir / "stereo4d-idx.json"
        map_json = wds_dir / "key_to_idx.json"
        
        print(f"loading wds index from {idx_json}...")
        t0 = time.perf_counter()
        self.idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
        print(f"loaded {len(self.idx_ds)} sequence samples in {time.perf_counter() - t0:.2f}s")
        
        with open(map_json) as f:
            self.key_to_idx = json.load(f)

        # Load metadata (keeping original logic but simplified)
        available_seqs = list(self.key_to_idx.keys())
        print(f"found {len(available_seqs)} sequences in WebDataset")

        # Load stats and metadata (same as original)
        stats_csv = Path(config.dataset.stereo4d.meta_dir) / "stats.csv"
        stats_df = pd.read_csv(stats_csv, skipinitialspace=True)

        meta_csv = Path(config.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
        meta_df = pd.read_csv(
            meta_csv,
            header=0,
            names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
        )

        # Process metadata efficiently
        available_seqs_set = set(available_seqs)
        merged = stats_df.merge(
            meta_df[['vid', 'clipid', 'timestamp']], 
            left_on=['ytid', 'clipid'], 
            right_on=['vid', 'clipid'], 
            how='inner'
        )
        merged = merged.groupby(['ytid', 'clipid']).first().reset_index()
        merged['seq'] = merged['ytid'] + '_' + merged['timestamp'].astype(int).astype(str)
        filtered = merged[merged['seq'].isin(available_seqs_set)]
        seq_to_frame_count = dict(zip(filtered['seq'], filtered['d_frame'].astype(int)))

        print(f"mapped {len(seq_to_frame_count)} sequences to frame counts")

        # Sample sequences
        available_with_counts = list(seq_to_frame_count.keys())
        limit = min(len(available_with_counts), config.data.len if not valid else config.data.valid_len)
        
        np.random.seed(config.seed)
        selected_seqs = np.random.choice(available_with_counts, size=limit, replace=False)

        # Build sequence list
        for seq in selected_seqs:
            self.sequence_paths.append(seq)
            self.frame_counts.append(seq_to_frame_count[seq])

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")
        
        # Precompute frame triplets
        self._compute_triplets()

    def _load_single_frame(self, seq: str, idx: int):
        """Load ONLY the specific frame needed - no caching, minimal memory."""
        # Get video sample from WebDataset
        sample = self.idx_ds[self.key_to_idx[seq]]
        video_data = sample[".video.mp4"]
        
        # Create video reader from bytes
        video_bytes = video_data.read() if hasattr(video_data, "read") else video_data
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        
        # Load ONLY the specific frame - VideoReader handles seeking efficiently
        frame = vr[idx].asnumpy()
        
        # Convert color space once
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Immediately release video reader memory
        del vr
        
        return frame_rgb

    def _load_single_frame_annotations(self, seq: str, idx: int):
        """Load and process annotations using original logic, return specific frame."""
        # Get annotation sample  
        sample = self.idx_ds[self.key_to_idx[seq]]
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        
        # Use original annotation processing logic
        lengths = ann_data['track_lengths']
        
        # Get number of frames from sequence (we don't have frames loaded, so infer from data)
        num_f = ann_data['track_indices'].max() + 1  # max frame index + 1
        
        ts = ann_data.get('timestamps')
        timestamps = ts if ts is not None else np.arange(num_f)
        shape = (len(lengths), num_f, 3)
        tracks = np.full(shape, np.nan, dtype=np.float32)
        
        row_idx = np.repeat(np.arange(lengths.shape[0]), lengths)
        col_idx = ann_data['track_indices']
        tracks[row_idx, col_idx] = ann_data['track_coordinates']
        
        valid = (~np.isnan(tracks[..., 0])).astype(np.float32)[..., None]
        pcs = np.concatenate([tracks, valid], axis=-1).astype(np.float32)
        
        # Return only the specific frame we need: pcs is (num_tracks, num_frames, 4)
        return pcs[:, idx, :]  # (num_tracks, 4)

    def _load_camera_data(self, seq: str, idx: int):
        """Load camera intrinsics and extrinsics for specific frame."""
        sample = self.idx_ds[self.key_to_idx[seq]]
        
        # Load intrinsics (same for all frames in sequence)
        intrinsics = np.load(sample[".intr.npy"], allow_pickle=True)
        
        # Load only the specific frame's extrinsics
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        extrinsics = geom.inv(ann_data['camera2world'][idx])
        
        return intrinsics, extrinsics

    def get_frame_info(self, seq: str, idx: int):
        """
        Ultra-optimized frame loading for single access patterns.
        
        Loads only the exact data needed with minimal memory usage.
        Perfect for large datasets where each frame is accessed once.
        
        Args:
            seq (str): sequence identifier
            idx (int): frame index
            
        Returns:
            dict with image, world_pc_valid, cam, dm, instance
        """
        # Load only the specific frame 
        img = self._load_single_frame(seq, idx)
        
        # Load only annotations for this frame
        world_pc = self._load_single_frame_annotations(seq, idx)
        
        # Get camera data for this frame
        intrinsics, extrinsics = self._load_camera_data(seq, idx)
        cam = (intrinsics, extrinsics)
        
        return dict(
            image=img,
            world_pc_valid=world_pc,
            cam=cam,
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )


import time, os
import hydra
from omegaconf import DictConfig, open_dict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# --- user-controlled setting ------------------------------------------------
# option 1 – edit this constant:
TARGET_SAMPLES = 10
# option 2 – override at runtime:  NUM_SAMPLES=25 poetry run python -m loaders.stereo4dv5
# TARGET_SAMPLES = int(os.environ.get("NUM_SAMPLES", TARGET_SAMPLES))
# ---------------------------------------------------------------------------

def add_batch_size_wrapper(batch):
    batch = default_collate(batch)
    batch["batch_size"] = len(batch["left_pm"])
    return batch

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    # replicate training-time patches so dataset builds
    with open_dict(config):
        config.data.len = config.train.iterations * config.data.batch_size
        config.data.valid_len = config.data.valid_len * config.data.batch_size

    # dataset + dataloader (identical settings to training loop)
    dataset = Stereo4Dv5(config, valid=False)
    dist = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if dist else 1
    sampler = DistributedSampler(dataset) if dist else None

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size // world,
        shuffle=(sampler is None),
        num_workers=config.data.num_workers,
        sampler=sampler,
        collate_fn=add_batch_size_wrapper,
    )

    # iterate and time
    seen = 0
    pbar = tqdm(total=TARGET_SAMPLES, desc="loading samples")
    t0 = time.perf_counter()
    for batch in loader:
        bs = batch["batch_size"]
        seen += bs
        pbar.update(bs if seen <= TARGET_SAMPLES else bs - (seen - TARGET_SAMPLES))
        if seen >= TARGET_SAMPLES:
            break
    pbar.close()
    print(f"pulled {min(seen, TARGET_SAMPLES)} samples in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()

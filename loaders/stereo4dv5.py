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

    def __init__(self, config, valid: bool = False, time_debug: bool = False):
        super().__init__(config, valid, time_debug)  # Pass time_debug to parent
        print("loading stereo4d v5 dataset...")

        self.dataset_label = config.dataset.stereo4d.name
        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        self.config = config
        self.time_debug = time_debug  # Store time_debug flag

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
        if self.time_debug:
            t0 = time.perf_counter()
        
        # Get video sample from WebDataset
        if self.time_debug:
            t1 = time.perf_counter()
        sample = self.idx_ds[self.key_to_idx[seq]]
        if self.time_debug:
            print(f"[TIME] Get sample from idx_ds: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        if self.time_debug:
            t1 = time.perf_counter()
        video_data = sample[".video.mp4"]
        if self.time_debug:
            print(f"[TIME] Access video data: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Create video reader from bytes
        if self.time_debug:
            t1 = time.perf_counter()
        video_bytes = video_data.read() if hasattr(video_data, "read") else video_data
        if self.time_debug:
            print(f"[TIME] Read video bytes: {(time.perf_counter() - t1)*1000:.2f}ms")
            
        if self.time_debug:
            t1 = time.perf_counter()
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        if self.time_debug:
            print(f"[TIME] Create VideoReader: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Load ONLY the specific frame - VideoReader handles seeking efficiently
        if self.time_debug:
            t1 = time.perf_counter()
        frame = vr[idx].asnumpy()
        if self.time_debug:
            print(f"[TIME] Load frame {idx}: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Convert color space once
        if self.time_debug:
            t1 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.time_debug:
            print(f"[TIME] Convert BGR to RGB: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Immediately release video reader memory
        del vr
        
        if self.time_debug:
            print(f"[TIME] Total _load_single_frame: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        return frame_rgb

    def _load_single_frame_annotations(self, seq: str, idx: int):
        """Load and process annotations using original logic, return specific frame."""
        if self.time_debug:
            t0 = time.perf_counter()
            
        # Get annotation sample  
        if self.time_debug:
            t1 = time.perf_counter()
        sample = self.idx_ds[self.key_to_idx[seq]]
        if self.time_debug:
            print(f"[TIME] Get sample for annotations: {(time.perf_counter() - t1)*1000:.2f}ms")
            
        if self.time_debug:
            t1 = time.perf_counter()
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        if self.time_debug:
            print(f"[TIME] Load ann.npz: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Use original annotation processing logic
        if self.time_debug:
            t1 = time.perf_counter()
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
        if self.time_debug:
            print(f"[TIME] Process annotations: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Return only the specific frame we need: pcs is (num_tracks, num_frames, 4)
        result = pcs[:, idx, :]  # (num_tracks, 4)
        
        if self.time_debug:
            print(f"[TIME] Total _load_single_frame_annotations: {(time.perf_counter() - t0)*1000:.2f}ms")
            
        return result

    def _load_camera_data(self, seq: str, idx: int):
        """Load camera intrinsics and extrinsics for specific frame."""
        if self.time_debug:
            t0 = time.perf_counter()
            
        if self.time_debug:
            t1 = time.perf_counter()
        sample = self.idx_ds[self.key_to_idx[seq]]
        if self.time_debug:
            print(f"[TIME] Get sample for camera: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Load intrinsics (same for all frames in sequence)
        if self.time_debug:
            t1 = time.perf_counter()
        intrinsics = np.load(sample[".intr.npy"], allow_pickle=True)
        if self.time_debug:
            print(f"[TIME] Load intrinsics: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        # Load only the specific frame's extrinsics
        if self.time_debug:
            t1 = time.perf_counter()
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        extrinsics = geom.inv(ann_data['camera2world'][idx])
        if self.time_debug:
            print(f"[TIME] Load and invert extrinsics: {(time.perf_counter() - t1)*1000:.2f}ms")
        
        if self.time_debug:
            print(f"[TIME] Total _load_camera_data: {(time.perf_counter() - t0)*1000:.2f}ms")
            
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
        if self.time_debug:
            total_start = time.perf_counter()
            print(f"\n[TIME] get_frame_info({seq}, {idx})")
            
        # Load only the specific frame 
        img = self._load_single_frame(seq, idx)
        
        # Load only annotations for this frame
        world_pc = self._load_single_frame_annotations(seq, idx)
        
        # Get camera data for this frame
        intrinsics, extrinsics = self._load_camera_data(seq, idx)
        cam = (intrinsics, extrinsics)
        
        if self.time_debug:
            print(f"[TIME] TOTAL get_frame_info: {(time.perf_counter() - total_start)*1000:.2f}ms")
        
        return dict(
            image=img,
            world_pc_valid=world_pc,
            cam=cam,
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )

    def get_frame_infos(self, seq: str, idxs: list):
        """
        Optimized batch loading of multiple frames from the same sequence.
        Opens files only once and loads all requested frames efficiently.
        
        Args:
            seq (str): sequence identifier
            idxs (list): list of frame indices to load
            
        Returns:
            list of dicts with image, world_pc_valid, cam, dm, instance for each frame
        """
        if self.time_debug:
            total_start = time.perf_counter()
            print(f"\n[TIME] get_frame_infos({seq}, {idxs})")
        
        # Get the sample once for all frames
        if self.time_debug:
            t0 = time.perf_counter()
        sample = self.idx_ds[self.key_to_idx[seq]]
        if self.time_debug:
            print(f"[TIME] Get sample: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Load video data and create VideoReader ONCE
        if self.time_debug:
            t0 = time.perf_counter()
        video_data = sample[".video.mp4"]
        video_bytes = video_data.read() if hasattr(video_data, "read") else video_data
        if self.time_debug:
            print(f"[TIME] Read video bytes: {(time.perf_counter() - t0)*1000:.2f}ms")
            
        if self.time_debug:
            t0 = time.perf_counter()
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        if self.time_debug:
            print(f"[TIME] Create VideoReader: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Load all frames from the video
        if self.time_debug:
            t0 = time.perf_counter()
        frames = []
        for idx in idxs:
            if self.time_debug:
                t1 = time.perf_counter()
            frame = vr[idx].asnumpy()
            if self.time_debug:
                print(f"[TIME]   Load frame {idx}: {(time.perf_counter() - t1)*1000:.2f}ms")
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if self.time_debug:
            print(f"[TIME] Load all {len(idxs)} frames: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Release video reader
        del vr
        
        # Load annotation data once
        if self.time_debug:
            t0 = time.perf_counter()
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        if self.time_debug:
            print(f"[TIME] Load ann.npz: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Load intrinsics once (same for all frames in sequence)
        if self.time_debug:
            t0 = time.perf_counter()
        intrinsics = np.load(sample[".intr.npy"], allow_pickle=True)
        if self.time_debug:
            print(f"[TIME] Load intrinsics: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Process annotations once
        if self.time_debug:
            t0 = time.perf_counter()
            print("[TIME] ── Process annotations ──")
        lengths = ann_data['track_lengths']
        
        if self.time_debug:
            t1 = time.perf_counter()
        frame_idxs = np.asarray(idxs)
        if self.time_debug:
            print(f"[TIME]     build frame_idxs: {(time.perf_counter()-t1)*1000:.2f}ms")
        
        if self.time_debug:
            t1 = time.perf_counter()
        col_of_frame = {f: j for j, f in enumerate(frame_idxs)}
        if self.time_debug:
            print(f"[TIME]     dict frame→col:  {(time.perf_counter()-t1)*1000:.2f}ms")
        
        if self.time_debug:
            t1 = time.perf_counter()
        col_idx_full = ann_data['track_indices']
        keep = np.isin(col_idx_full, frame_idxs)
        if self.time_debug:
            print(f"[TIME]     mask np.isin:     {(time.perf_counter()-t1)*1000:.2f}ms "
                  f"({keep.sum()} / {keep.size} rows kept)")
        
        # fast track-id lookup without np.repeat
        if keep.any():
            if self.time_debug:
                t1 = time.perf_counter()
            obs_idx = np.flatnonzero(keep)
            if self.time_debug:
                print(f"[TIME]       np.flatnonzero: {(time.perf_counter()-t1)*1000:.2f}ms")
            
            if self.time_debug:
                t1 = time.perf_counter()
            track_ends = lengths.cumsum()          # exclusive ends
            row_s = np.searchsorted(track_ends, obs_idx, side='right')
            if self.time_debug:
                print(f"[TIME]       searchsorted:  {(time.perf_counter()-t1)*1000:.2f}ms")
            
            if self.time_debug:
                t1 = time.perf_counter()
            frames_kept = col_idx_full[keep]
            # vectorised mapping frame id -> column
            max_frame = int(col_idx_full.max())
            frame2col = np.full(max_frame + 1, -1, dtype=np.int32)
            frame2col[frame_idxs] = np.arange(len(frame_idxs), dtype=np.int32)
            col_s = frame2col[frames_kept]
            coord_s = ann_data['track_coordinates'][keep]
            if self.time_debug:
                print(f"[TIME]       build col/coords: {(time.perf_counter()-t1)*1000:.2f}ms")
        else:
            row_s = col_s = coord_s = np.empty((0,), dtype=np.int32)
        
        if self.time_debug:
            t1 = time.perf_counter()
        shape_sel = (len(lengths), len(frame_idxs), 3)
        tracks = np.full(shape_sel, np.nan, dtype=np.float32)
        if self.time_debug:
            print(f"[TIME]     allocate tracks: {(time.perf_counter()-t1)*1000:.2f}ms "
                  f"(shape={shape_sel})")
        
        if len(row_s):
            if self.time_debug:
                t1 = time.perf_counter()
            tracks[row_s, col_s] = coord_s
            if self.time_debug:
                print(f"[TIME]     scatter coords: {(time.perf_counter()-t1)*1000:.2f}ms")
        
        if self.time_debug:
            t1 = time.perf_counter()
        valid = (~np.isnan(tracks[..., 0])).astype(np.float32)[..., None]
        pcs = np.concatenate([tracks, valid], axis=-1).astype(np.float32)
        if self.time_debug:
            print(f"[TIME]     build mask+pcs: {(time.perf_counter()-t1)*1000:.2f}ms")
        
        if self.time_debug:
            print(f"[TIME] ── Process annotations total: {(time.perf_counter()-t0)*1000:.2f}ms")
        
        # Get all extrinsics we need
        if self.time_debug:
            t0 = time.perf_counter()
        extrinsics_list = [geom.inv(ann_data['camera2world'][idx]) for idx in idxs]
        if self.time_debug:
            print(f"[TIME] Compute extrinsics: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        # Build results
        if self.time_debug:
            t0 = time.perf_counter()
        results = []
        for i, idx in enumerate(idxs):
            results.append(dict(
                image=frames[i],
                world_pc_valid=pcs[:, col_of_frame[idx], :],  # (num_tracks, 4) for this frame
                cam=(intrinsics, extrinsics_list[i]),
                dm=None,
                instance=f"{seq}_{idx:05d}",
            ))
        if self.time_debug:
            print(f"[TIME] Build results: {(time.perf_counter() - t0)*1000:.2f}ms")
        
        if self.time_debug:
            print(f"[TIME] TOTAL get_frame_infos: {(time.perf_counter() - total_start)*1000:.2f}ms")
        
        return results


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

    # Enable time_debug for detailed timing
    time_debug = True
    
    # dataset + dataloader (identical settings to training loop)
    dataset = Stereo4Dv5(config, valid=False, time_debug=time_debug)
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


'''

implementations:

def naive_pcs(lengths, track_indices, track_coordinates, idxs):
    num_f = track_indices.max() + 1
    n_tracks = len(lengths)
    row_idx = np.repeat(np.arange(n_tracks), lengths)
    tracks = np.full((n_tracks, num_f, 3), np.nan, np.float32)
    tracks[row_idx, track_indices] = track_coordinates
    valid = (~np.isnan(tracks[..., 0]))[..., None].astype(np.float32)
    pcs = np.concatenate([tracks, valid], axis=-1)
    # slice the columns we care about
    sel = np.stack([pcs[:, idx, :] for idx in idxs], axis=1)
    return sel

def optimized_pcs(lengths, track_indices, track_coordinates, idxs):
    frame_idxs = np.asarray(idxs)
    col_idx_full = track_indices
    keep = np.isin(col_idx_full, frame_idxs)

    if keep.any():
        obs_idx = np.flatnonzero(keep)
        row_s = np.searchsorted(lengths.cumsum(), obs_idx, side='right')

        frames_kept = col_idx_full[keep]
        max_frame = int(col_idx_full.max())
        frame2col = np.full(max_frame + 1, -1, dtype=np.int32)
        frame2col[frame_idxs] = np.arange(len(frame_idxs), dtype=np.int32)
        col_s = frame2col[frames_kept]

        coord_s = track_coordinates[keep]
    else:
        row_s = col_s = coord_s = np.empty((0,), dtype=np.int32)

    tracks = np.full((len(lengths), len(frame_idxs), 3), np.nan, np.float32)
    if len(row_s):
        tracks[row_s, col_s] = coord_s
    valid = (~np.isnan(tracks[..., 0]))[..., None].astype(np.float32)
    return np.concatenate([tracks, valid], axis=-1)

'''
"""
Author: Assistant
Date: 2025-05-23
Description: Stereo4Dv3 dataloader that loads sequence-level WebDataset samples
"""

import os, json, io, time, tempfile
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import webdataset as wds, wids
from decord import VideoReader, cpu

from .stereo_motion_base import StereoMotionBase
from .utils import geometry as geom


def _load_npy(blob):
    """Load numpy array from bytes or file."""
    if isinstance(blob, np.ndarray):
        return blob
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(bytes(blob)), allow_pickle=True)
    return np.load(blob, allow_pickle=True)


class Stereo4Dv4(StereoMotionBase):
    """
    Stereo-4D dataloader backed by sequence-level WebDataset shards.
    
    Each WebDataset sample contains:
    - Full video file (.mp4)
    - Full annotation file (.npz)
    - Camera intrinsics (.npy)
    """

    def __init__(self, config, valid: bool = False):
        super().__init__(config)
        print("loading stereo4d v3 dataset (sequence-level WebDataset)...")

        self.dataset_label = config.dataset.stereo4d.name

        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        self.config = config

        # WebDataset setup
        wds_dir = Path(config.dataset.stereo4d.path) / "wds" / split
        idx_json = wds_dir / "stereo4d-idx.json"
        map_json = wds_dir / "key_to_idx.json"
        
        print(f"loading wds index from {idx_json}...")
        t0 = time.perf_counter()
        self.idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
        print(f"loaded {len(self.idx_ds)} sequence samples in {time.perf_counter() - t0:.2f}s")
        
        with open(map_json) as f:
            self.key_to_idx = json.load(f)

        # Get available sequences directly from saved keys
        available_seqs = list(self.key_to_idx.keys())
        print(f"found {len(available_seqs)} sequences in WebDataset")

        # Load stats CSV only to get frame counts for available sequences
        stats_csv = Path(config.dataset.stereo4d.meta_dir) / "stats.csv"
        stats_df = pd.read_csv(stats_csv, skipinitialspace=True)

        # Load metadata CSV to map sequence names to ytid+clipid
        meta_csv = Path(config.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
        meta_df = pd.read_csv(
            meta_csv,
            header=0,
            names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
        )

        # Convert to set for O(1) lookups
        available_seqs_set = set(available_seqs)

        # Merge DataFrames instead of nested loops
        merged = stats_df.merge(
            meta_df[['vid', 'clipid', 'timestamp']], 
            left_on=['ytid', 'clipid'], 
            right_on=['vid', 'clipid'], 
            how='inner'
        )

        # Group by ytid+clipid and take first timestamp (matches original ts.values[0] behavior)
        merged = merged.groupby(['ytid', 'clipid']).first().reset_index()

        # Create sequence names and filter by available sequences
        merged['seq'] = merged['ytid'] + '_' + merged['timestamp'].astype(int).astype(str)
        filtered = merged[merged['seq'].isin(available_seqs_set)]

        # Create the mapping
        seq_to_frame_count = dict(zip(filtered['seq'], filtered['d_frame'].astype(int)))

        print(f"mapped {len(seq_to_frame_count)} sequences to frame counts")

        # Sample sequences from available ones
        available_with_counts = list(seq_to_frame_count.keys())
        limit = min(len(available_with_counts), config.data.len if not valid else config.data.valid_len)
        
        # Use numpy for consistent random sampling
        np.random.seed(config.seed)
        selected_seqs = np.random.choice(available_with_counts, size=limit, replace=False)

        # Build sequence list using selected sequences
        for seq in selected_seqs:
            self.sequence_paths.append(seq)
            self.frame_counts.append(seq_to_frame_count[seq])

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")
        
        # Precompute frame triplets
        self._compute_triplets()
        
        # Cache for loaded sequences
        self._seq_cache = {}
        self._cache_size = 10  # Keep last N sequences in memory

    def _load_sequence(self, seq: str):
        """Load and cache entire sequence data."""
        # Get sequence sample from WebDataset
        sample = self.idx_ds[self.key_to_idx[seq]]
        
        # Load components (note the dot prefix for WebDataset keys)
        video_data = sample[".video.mp4"]
        ann_data = np.load(sample[".ann.npz"], allow_pickle=True)
        intrinsics = _load_npy(sample[".intr.npy"])
        
        # Read video using decord directly from bytes
        video_bytes = video_data.read() if hasattr(video_data, "read") else video_data
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        
        frames = [vr[i].asnumpy() for i in range(len(vr))]

        
        # process annotations using optimized method
        lengths = ann_data['track_lengths']
        # use actual number of frames for timestamps default
        num_f = len(frames)
        ts = ann_data.get('timestamps')
        timestamps = ts if ts is not None else np.arange(num_f)
        shape = (len(lengths), num_f, 3)
        tracks = np.full(shape, np.nan, dtype=np.float32)
        
        row_idx = np.repeat(np.arange(lengths.shape[0]), lengths)
        col_idx = ann_data['track_indices']
        tracks[row_idx, col_idx] = ann_data['track_coordinates']
        
        valid = (~np.isnan(tracks[..., 0])).astype(np.float32)[..., None]
        pcs = np.concatenate([tracks, valid], axis=-1).astype(np.float32)
        
        # camera extrinsics
        extrinsics = ann_data['camera2world']
        
        return {
            'frames': frames,
            'pcs': pcs,  # (num_tracks, num_frames, 4)
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'num_frames': len(frames)
        }

    def get_frame_info(self, seq: str, idx: int):
        """
        Load image and point cloud for a specific sequence and frame.
        
        Args:
            seq (str): sequence identifier
            idx (int): frame index
            
        Returns:
            dict with image, world_pc_valid, cam, dm, instance
        """
        seq_data = self._load_sequence(seq)
        
        img = seq_data['frames'][idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr→rgb
        
        world_pc = seq_data['pcs'][:, idx, :]  # (num_tracks, 4)
        intrinsics = seq_data['intrinsics']
        extrinsics = geom.inv(seq_data['extrinsics'][idx])
        cam = (intrinsics, extrinsics)
        
        return dict(
            image=img,
            world_pc_valid=world_pc,
            cam=cam,
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )


import hydra
from omegaconf import DictConfig, open_dict
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_sequence(seq, dataset):
    try:
        # try loading first three frames
        dataset.get_frame_info(seq, 0)
        dataset.get_frame_info(seq, 1)
        dataset.get_frame_info(seq, 2)
        return seq, None
    except Exception as e:
        return seq, e

# usage:
# python -m loaders.stereo4d dataset.stereo4d.split=test
# python -m loaders.stereo4d dataset.stereo4d.split=test dataset.stereo4d.max_frame_window=30
# /scratch/projects/fouheylab/km6748/stereo4d-data/lefteye-perspective/train/yAAaVncw5g0_152118785-left_rectified.mp4
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    import torch
    import loaders.utils.viz as viz

    dataset_tracks = Stereo4Dv4(config)
    sequences = dataset_tracks.sequence_paths
    total = len(sequences)
    print(f"total triplets: {total}")

    max_workers = os.cpu_count() * 2  # number of threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # schedule all checks
        futures = [executor.submit(check_sequence, seq, dataset_tracks) for seq in sequences]
        for future in tqdm(as_completed(futures), total=total):
            seq, err = future.result()
            if err:
                print(f"Error loading sequence {seq}: {err}")

    for sample_idx in range(0, 4):
        sample_tracks = dataset_tracks[sample_idx]

        # print basic info about the triplet
        print(
            f"triplet frames: instance={sample_tracks['left_instance']}, {sample_tracks['right_instance']}| idxs={sample_tracks['idxs']}"
        )
        print("------------------------")
        for k, v in sample_tracks.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size(), v.dtype)
            else:
                print(k)

        # prepare point maps and images for visualization
        track_pms = np.array(
            [
                sample_tracks["left_pm"],
                sample_tracks["mid_pm"],
                sample_tracks["right_pm"],
            ]
        )

        c1 = sample_tracks["cam"]  # or sample_tracks["cam"]
        c2 = sample_tracks["cam_mid"]
        c3 = sample_tracks["cam_right"]

        img_l = sample_tracks["left_image"].permute(1, 2, 0).numpy()
        img_m = sample_tracks["mid_image"].permute(1, 2, 0).numpy()
        img_r = sample_tracks["right_image"].permute(1, 2, 0).numpy()

        images = [
            img_l,
            geom.recolor(track_pms[1], c1, c2, img_m),
            geom.recolor(track_pms[2], c1, c3, img_r),
        ]

        viz.visualize_image(
            sample_tracks["left_image"].permute(1, 2, 0).numpy(),
            name=f"{sample_idx}-left_image",
        )
        # viz.visualize_dm(sample_tracks['left_dm'], name=f"{sample_idx}-left_dm")
        viz.visualize_image(
            sample_tracks["mid_image"].permute(1, 2, 0).numpy(),
            name=f"{sample_idx}-mid_image",
        )
        # viz.visualize_dm(sample_tracks['mid_dm'], name=f"{sample_idx}-mid_dm")
        viz.visualize_image(
            sample_tracks["right_image"].permute(1, 2, 0).numpy(),
            name=f"{sample_idx}-right_image",
        )
        # viz.visualize_dm(sample_tracks['right_dm'], name=f"{sample_idx}-right_dm")

        # viz.visualize_pm(
        #     track_pms[0],
        #     image=images[0],
        #     cam=sample_tracks["cam"],
        #     name=f"{sample_idx}-left_pm",
        # )
        # viz.visualize_pm(
        #     track_pms[1],
        #     image=images[1],
        #     cam=sample_tracks["cam"],
        #     name=f"{sample_idx}-mid_pm",
        # )
        # viz.visualize_pm(
        #     track_pms[2],
        #     image=images[2],
        #     cam=sample_tracks["cam"],
        #     name=f"{sample_idx}-right_pm",
        # )

        # prepare motion maps
        left_to_mid = sample_tracks["left_to_mid_motion"]
        right_to_mid = sample_tracks["right_to_mid_motion"]

        # create single-step motion maps for each visualization
        h, w, _ = sample_tracks["left_pm"].shape

        # 1. tracks + left_to_mid
        left_to_mid_map = np.zeros((1, h, w, 4), dtype=np.float32)
        left_to_mid_map[0] = left_to_mid

        # 2. tracks + right_to_mid
        right_to_mid_map = np.zeros((1, h, w, 4), dtype=np.float32)
        right_to_mid_map[0] = right_to_mid

        # viz.visualize_sequence_from_pms(
        #     track_pms[:2],  # just left and mid for left_to_mid
        #     left_to_mid_map,
        #     images[:2],
        #     name=f"{sample_idx}-tracksl",
        # )

        # viz.visualize_sequence_from_pms(
        #     np.array([track_pms[2], track_pms[1]]),  # right and mid for right_to_mid
        #     right_to_mid_map,
        #     [images[2], images[1]],
        #     name=f"{sample_idx}-tracksr",
        # )


if __name__ == "__main__":
    main()
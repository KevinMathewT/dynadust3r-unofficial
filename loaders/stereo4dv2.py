"""
Author: Kevin Mathew T
Date: 2025-05-20
"""

import os, json, io, math, random
from pathlib import Path

import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import webdataset as wds, wids  # wds≙reading, wids≙random-access

from .stereo_motion_base import StereoMotionBase
import utils.geometry as geom  # has inv()

# -----------------------------------------------------------------------------


def _load_npy(blob):  # small helper
    if isinstance(blob, np.ndarray):
        return blob
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(bytes(blob)), allow_pickle=True)
    return np.load(blob, allow_pickle=True)


class Stereo4Dv2(StereoMotionBase):
    """
    Stereo-4D dataloader backed by WebDataset shards.

    one key  → one frame
    keys     → "<ytid>_<timestamp>_<frame:05d>"
    sample   → {'.rgb.npy', '.pc.npy', '.intr.npy', '.extr.npy'}
    """

    def __init__(self, config, valid: bool = False):
        super().__init__(config)
        print("loading stereo4d-wds dataset …")

        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window

        # ---------- sequence discovery identical to v2 ----------
        meta_csv = (
            Path(config.dataset.stereo4d.meta_dir)
            / "stereo4d_id_to_time_and_fov_metadata.csv"
        )
        stats_csv = Path(config.dataset.stereo4d.meta_dir) / "stats.csv"
        meta_df = pd.read_csv(
            meta_csv,
            header=0,
            names=[
                "vid",
                "clipid",
                "timestamp",
                "start_yaw",
                "end_yaw",
                "start_tilt",
                "end_tilt",
            ],
        )

        stats_df = pd.read_csv(stats_csv, skipinitialspace=True)
        stats_df = stats_df[stats_df["displacement_percentage_50"] > 0.10]
        stats_df = stats_df[stats_df["d_frame"] > 5 * 16]

        # ───────── additional split‑based filtering (parallel) ────────────
        def _has_mp4(row):
            vid, cid = row["ytid"], row["clipid"]
            ts = meta_df.loc[(meta_df.vid == vid) & (meta_df.clipid == cid), "timestamp"]
            if ts.empty:
                return False
            seq = f"{vid}_{int(ts.values[0])}"
            return (Path(config.dataset.stereo4d.lefteye_dir) / split / f"{seq}-left_rectified.mp4").exists()

        exist_mask = Parallel(n_jobs=os.cpu_count())(delayed(_has_mp4)(row) for _, row in stats_df.iterrows())
        stats_df   = stats_df[exist_mask]
        # ──────────────────────────────────────────────────────────────────

        limit = min(len(stats_df), config.data.len)
        stats_df = stats_df.sample(n=limit, random_state=config.seed)

        for _, r in stats_df.iterrows():
            ts = meta_df.loc[
                (meta_df.vid == r["ytid"]) & (meta_df.clipid == r["clipid"]),
                "timestamp",
            ]
            if ts.empty:
                continue
            seq = f"{r['ytid']}_{int(ts.values[0])}"
            self.sequence_paths.append(seq)
            self.frame_counts.append(int(r["d_frame"]))

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")
        self._compute_triplets()

        # ---------- wds random-access ----------
        wds_dir = Path(config.dataset.stereo4d.path) / "wds" / split  # “…/wds/train”
        idx_json = wds_dir / "stereo4d-idx.json"
        map_json = wds_dir / "key_to_idx.json"
        
        print(f"loading wds index from {idx_json} …")
        t0 = time.perf_counter()
        self.idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
        print(f"loaded {len(self.idx_ds)} samples in {time.perf_counter() - t0:.2f}s")
        with open(map_json) as f:
            self.key_to_idx = json.load(f)

        # breakpoint()

    # ------------------------------------------------------------------
    # helpers
    def _get_sample(self, seq: str, idx: int):
        key = f"{seq}_{idx:05d}"
        sample = self.idx_ds[self.key_to_idx[key]]
        rgb = _load_npy(sample.get(".rgb.npy"))  # (H,W,3) float32
        pc = _load_npy(sample.get(".pc.npy"))  # (N,4)  float32
        intr = _load_npy(sample.get(".intr.npy"))  # (3,3)  float32
        extr = _load_npy(sample.get(".extr.npy"))  # (4,4)  float32
        return rgb, pc, intr, extr, key

    # ------------------------------------------------------------------
    # required by StereoMotionBase
    def get_frame_info(self, seq: str, idx: int):
        rgb, pc, K, E, instance = self._get_sample(seq, idx)
        return dict(
            image=rgb,  # (H,W,3) float32 rgb
            world_pc_valid=pc,  # (N,4)
            cam=(K, E),  # (3,3), (4,4)
            dm=None,
            instance=instance,
        )


import hydra
from omegaconf import DictConfig


# usage:
# python -m loaders.stereo4dv2 dataset.stereo4d.split=test
# python -m loaders.stereo4dv2 dataset.stereo4d.split=test dataset.stereo4d.max_frame_window=30
# /scratch/projects/fouheylab/km6748/stereo4d-data/lefteye-perspective/train/yAAaVncw5g0_152118785-left_rectified.mp4
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    import torch
    import loaders.utils.viz as viz

    dataset_tracks = Stereo4Dv2(config)
    print(f"total triplets: {len(dataset_tracks)}")

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

        viz.visualize_pm(
            track_pms[0],
            image=images[0],
            cam=sample_tracks["cam"],
            name=f"{sample_idx}-left_pm",
        )
        viz.visualize_pm(
            track_pms[1],
            image=images[1],
            cam=sample_tracks["cam"],
            name=f"{sample_idx}-mid_pm",
        )
        viz.visualize_pm(
            track_pms[2],
            image=images[2],
            cam=sample_tracks["cam"],
            name=f"{sample_idx}-right_pm",
        )

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

        viz.visualize_sequence_from_pms(
            track_pms[:2],  # just left and mid for left_to_mid
            left_to_mid_map,
            images[:2],
            name=f"{sample_idx}-tracksl",
        )

        viz.visualize_sequence_from_pms(
            np.array([track_pms[2], track_pms[1]]),  # right and mid for right_to_mid
            right_to_mid_map,
            [images[2], images[1]],
            name=f"{sample_idx}-tracksr",
        )


if __name__ == "__main__":
    main()

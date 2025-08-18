# stereo4dv6.py

import os, io, time
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

from .stereo_motion_base import StereoMotionBase
import utils.geometry as geom


def _intrinsic_K(width: int, hfov_deg: float) -> np.ndarray:
    # compute pinhole intrinsics from hfov and image width
    fx = width * 0.5 / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([[fx, 0, width / 2],
                     [0, fx, width / 2],
                     [0,  0,          1]], dtype=np.float32)  # (3, 3)


def _optimized_pcs(lengths: np.ndarray,
                   track_indices: np.ndarray,
                   track_coordinates: np.ndarray,
                   idxs: np.ndarray) -> np.ndarray:
    # vectorized scatter into (tracks, len(idxs), 3), with valid mask appended
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
        row_s = col_s = np.empty((0,), dtype=np.int32)
        coord_s = np.empty((0, 3), dtype=np.float32)

    tracks = np.full((len(lengths), len(frame_idxs), 3), np.nan, np.float32)  # (T, F, 3)
    if len(row_s):
        tracks[row_s, col_s] = coord_s
    valid = (~np.isnan(tracks[..., 0]))[..., None].astype(np.float32)  # (T, F, 1)
    return np.concatenate([tracks, valid], axis=-1)  # (T, F, 4)


class Stereo4D(StereoMotionBase):
    """
    ultra-lightweight Stereo-4D dataloader (direct-from-disk).
    reads sequence-level mp4/npz files without webdataset.

    design:
    - single-frame path opens & seeks once, no persistent caches
    - multi-frame path opens once per sequence, batches seeks
    - intrinsics computed from hfov + width (memoized per seq)
    """

    def __init__(self, config, valid: bool = False, time_debug: bool = False):
        super().__init__(config, valid, time_debug)
        print("loading stereo4d dataset (direct-from-disk)...")

        self.dataset_label = config.dataset.stereo4d.name
        self.split = config.dataset.stereo4d.valid_split if valid else config.dataset.stereo4d.train_split
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        
        self.config = config
        self.time_debug = time_debug

        # dirs
        self.left_mp4_dir = Path(config.dataset.stereo4d.lefteye_dir) / self.split
        self.npz_dir = Path(config.dataset.stereo4d.path) / self.split

        # lightweight per-seq memo for intrinsics width
        self._intrinsics_by_seq: Dict[str, np.ndarray] = {}

        # build available sequences + frame counts using provided metadata
        stats_csv = Path(config.dataset.stereo4d.meta_dir) / "stats.csv"
        meta_csv = Path(config.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"

        stats_df = pd.read_csv(stats_csv, skipinitialspace=True)
        meta_df = pd.read_csv(
            meta_csv,
            header=0,
            names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
        )

        merged = stats_df.merge(
            meta_df[["vid", "clipid", "timestamp"]],
            left_on=["ytid", "clipid"],
            right_on=["vid", "clipid"],
            how="inner",
        )
        merged = merged.groupby(["ytid", "clipid"]).first().reset_index()
        merged["seq"] = merged["ytid"] + "_" + merged["timestamp"].astype(int).astype(str)

        # keep only seqs that have both mp4 + npz on disk
        def _exists(seq: str) -> bool:
            mp4 = self.left_mp4_path(seq)
            npz = self.npz_path(seq)
            return mp4.is_file() and npz.is_file()

        merged = merged[merged["seq"].map(_exists)]
        seq_to_frame_count = dict(zip(merged["seq"], merged["d_frame"].astype(int)))

        print(f"found {len(seq_to_frame_count)} sequences on disk")

        # sample sequences
        available = list(seq_to_frame_count.keys())
        limit = min(len(available), config.data.len if not valid else config.data.valid_len)

        np.random.seed(config.seed)
        selected = np.random.choice(available, size=limit, replace=False)

        for seq in selected:
            self.sequence_paths.append(seq)
            self.frame_counts.append(int(seq_to_frame_count[seq]))

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")

        # precompute triplets as in base impl
        self._compute_triplets()

    # ----------------------------- path helpers -----------------------------

    def left_mp4_path(self, seq: str) -> Path:
        return self.left_mp4_dir / f"{seq}-left_rectified.mp4"

    def npz_path(self, seq: str) -> Path:
        return self.npz_dir / f"{seq}.npz"

    # ----------------------------- core loaders -----------------------------

    def _load_single_frame(self, seq: str, idx: int) -> np.ndarray:
        if self.time_debug:
            t0 = time.perf_counter()
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))
        frame = vr[idx].asnumpy()  # (H, W, 3)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3)
        del vr
        if self.time_debug:
            print(f"[TIME] single frame {seq}[{idx}]: {(time.perf_counter()-t0)*1000:.2f}ms")
        return frame_rgb  # (H, W, 3)

    def _load_single_frame_annotations(self, seq: str, idx: int) -> np.ndarray:
        if self.time_debug:
            t0 = time.perf_counter()
        ann = np.load(self.npz_path(seq), allow_pickle=True)
        pcs_all = _optimized_pcs(
            lengths=ann["track_lengths"],
            track_indices=ann["track_indices"],
            track_coordinates=ann["track_coordinates"],
            idxs=np.array([idx], dtype=np.int32),
        )  # (T, 1, 4)
        pcs = pcs_all[:, 0, :]  # (T, 4)
        if self.time_debug:
            print(f"[TIME] ann frame {seq}[{idx}]: {(time.perf_counter()-t0)*1000:.2f}ms")
        return pcs.astype(np.float32)  # (T, 4)

    def _get_intrinsics(self, seq: str) -> np.ndarray:
        K = self._intrinsics_by_seq.get(seq)
        if K is not None:
            return K  # (3, 3)
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))
        h, w = vr[0].shape[0], vr[0].shape[1]
        del vr
        K = _intrinsic_K(w, self.hfov)  # (3, 3)
        self._intrinsics_by_seq[seq] = K
        return K  # (3, 3)

    def _load_camera_data(self, seq: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.time_debug:
            t0 = time.perf_counter()
        K = self._get_intrinsics(seq)  # (3, 3)
        ann = np.load(self.npz_path(seq), allow_pickle=True)
        extr = geom.inv(ann["camera2world"][idx])  # (4, 4) or (3, 4) depending on your utils
        if self.time_debug:
            print(f"[TIME] cam {seq}[{idx}]: {(time.perf_counter()-t0)*1000:.2f}ms")
        return K, extr  # (3, 3), (4, 4)

    # ------------------------------ public api ------------------------------

    def get_frame_info(self, seq: str, idx: int) -> dict:
        if self.time_debug:
            t0 = time.perf_counter()
            print(f"\n[TIME] get_frame_info({seq}, {idx})")

        img = self._load_single_frame(seq, idx)  # (H, W, 3)
        world_pc = self._load_single_frame_annotations(seq, idx)  # (T, 4)
        intr, extr = self._load_camera_data(seq, idx)  # (3, 3), (4, 4)

        if self.time_debug:
            print(f"[TIME] TOTAL get_frame_info: {(time.perf_counter()-t0)*1000:.2f}ms")

        return dict(
            image=img,                               # (H, W, 3)
            world_pc_valid=world_pc,                 # (T, 4)
            cam=(intr, extr),                        # ((3,3), (4,4))
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )

    def get_frame_infos(self, seq: str, idxs: List[int]) -> List[dict]:
        if self.time_debug:
            t0 = time.perf_counter()
            print(f"\n[TIME] get_frame_infos({seq}, {idxs})")

        # open video once
        t1 = time.perf_counter()
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))
        if self.time_debug:
            print(f"[TIME] create VR: {(time.perf_counter()-t1)*1000:.2f}ms")

        # batch load frames
        frames: List[np.ndarray] = []
        t1 = time.perf_counter()
        for i in idxs:
            frames.append(vr[i].asnumpy())  # (H, W, 3)
        if self.time_debug:
            print(f"[TIME] read {len(idxs)} frames: {(time.perf_counter()-t1)*1000:.2f}ms")
        del vr

        # convert color space once
        t1 = time.perf_counter()
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]  # each (H, W, 3)
        if self.time_debug:
            print(f"[TIME] bgr→rgb {len(frames)}: {(time.perf_counter()-t1)*1000:.2f}ms")

        # load ann once
        t1 = time.perf_counter()
        ann = np.load(self.npz_path(seq), allow_pickle=True)
        if self.time_debug:
            print(f"[TIME] load ann: {(time.perf_counter()-t1)*1000:.2f}ms")

        # intrinsics once
        intr = self._get_intrinsics(seq)  # (3, 3)

        # annotations for selected frames
        t1 = time.perf_counter()
        pcs_all = _optimized_pcs(
            lengths=ann["track_lengths"],
            track_indices=ann["track_indices"],
            track_coordinates=ann["track_coordinates"],
            idxs=np.asarray(idxs, dtype=np.int32),
        )  # (T, F, 4)
        if self.time_debug:
            print(f"[TIME] build pcs: {(time.perf_counter()-t1)*1000:.2f}ms")

        # extrinsics for selected frames
        t1 = time.perf_counter()
        c2w = ann["camera2world"]
        extr_list = [geom.inv(c2w[i]) for i in idxs]  # list of (4, 4)
        if self.time_debug:
            print(f"[TIME] extrinsics: {(time.perf_counter()-t1)*1000:.2f}ms")

        # col lookup
        col_of_frame = {f: j for j, f in enumerate(idxs)}

        # package
        t1 = time.perf_counter()
        out: List[dict] = []
        for j, fidx in enumerate(idxs):
            out.append(dict(
                image=frames[j],                                    # (H, W, 3)
                world_pc_valid=pcs_all[:, col_of_frame[fidx], :],   # (T, 4)
                cam=(intr, extr_list[j]),                           # ((3,3), (4,4))
                dm=None,
                instance=f"{seq}_{fidx:05d}",
            ))
        if self.time_debug:
            print(f"[TIME] build results: {(time.perf_counter()-t1)*1000:.2f}ms")
            print(f"[TIME] TOTAL get_frame_infos: {(time.perf_counter()-t0)*1000:.2f}ms")

        return out

"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .stereo_motion_base import StereoMotionBase
import utils.geometry as geom


class PointOdyssey(StereoMotionBase):
    def __init__(self, config, valid: bool = False, time_debug: bool = False):
        """
        pointodyssey dataset implementation
        """
        super().__init__(config, valid, time_debug)
        print("loading pointodyssey dataset...")

        self.dataset_label = config.dataset.pointodyssey.name
        self.dataset_location = config.dataset.pointodyssey.path
        self.max_frame_window = config.dataset.pointodyssey.max_frame_window

        self.split = (
            config.dataset.pointodyssey.valid_split if valid
            else config.dataset.pointodyssey.train_split
        )

        split_dir = os.path.join(self.dataset_location, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"dataset split directory not found: {split_dir}")

        # collect sequences that have the expected structure
        for item in sorted(os.listdir(split_dir)):
            seq_path = os.path.join(split_dir, item)
            if (
                os.path.isdir(seq_path)
                and os.path.exists(os.path.join(seq_path, "rgbs"))
                and os.path.exists(os.path.join(seq_path, "depths"))
                and os.path.exists(os.path.join(seq_path, "anno.npz"))
            ):
                self.sequence_paths.append(seq_path)

        # count frames by enumerating jpgs (sorted numerically)
        self.frame_counts = []
        for seq_path in self.sequence_paths:
            rgb_dir = os.path.join(seq_path, "rgbs")
            jpgs = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            if not jpgs:
                self.frame_counts.append(0)
                continue
            # ensure numeric sort in case of varying zero-padding
            jpgs.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            self.frame_counts.append(len(jpgs))

        # precompute triplets in the base
        self._compute_triplets()

    # ------------------------- file helpers -------------------------

    @staticmethod
    def _rgb_path(sequence_path: str, idx: int) -> str:
        return f"{sequence_path}/rgbs/rgb_{idx:05d}.jpg"

    @staticmethod
    def _depth_path(sequence_path: str, idx: int) -> str:
        return f"{sequence_path}/depths/depth_{idx:05d}.png"

    @staticmethod
    def _read_rgb_bgr_to_rgb(path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"missing image: {path}")
        return img_bgr[:, :, ::-1]  # (H, W, 3)  bgr->rgb view  # (H, W, 3)

    @staticmethod
    def _read_depth_to_meters(path: str) -> np.ndarray:
        dm_u16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if dm_u16 is None:
            raise FileNotFoundError(f"missing depth: {path}")
        dm = dm_u16.astype(np.float32) / 65535.0 * 1000.0  # dataset-specific scaling  # (H, W)
        return dm

    # ------------------------- single frame api -------------------------

    def get_frame_info(self, sequence_path: str, frame_index: int):
        """
        returns a single frame pack  # image: (H, W, 3), world_pc_valid: (N, 4)
        """
        # io
        image = self._read_rgb_bgr_to_rgb(self._rgb_path(sequence_path, frame_index))  # (H, W, 3)
        dm = self._read_depth_to_meters(self._depth_path(sequence_path, frame_index))  # (H, W)

        # annotations
        ann = np.load(f"{sequence_path}/anno.npz", allow_pickle=True, mmap_mode="r")
        intrinsics = ann["intrinsics"][frame_index]       # (3, 3)
        extrinsics = ann["extrinsics"][frame_index]       # (4, 4)
        cam = (intrinsics, extrinsics)                    # ((3,3), (4,4))

        world_pc = ann["trajs_3d"][frame_index]           # (N, 3)
        visibs = ann["visibs"] if "visibs" in ann.files else ann["valids"]
        validity = visibs[frame_index][..., None].astype(np.float32)  # (N, 1)
        world_pc_valid = np.concatenate([world_pc, validity], axis=1).astype(np.float32)  # (N, 4)

        # optional crop
        if self.config.data.crop:
            out_res = (self.config.data.size, self.config.data.size)  # (W, H)
            image, dm, cam = self.crop_data(image, dm, cam, out_res)  # apply cropping

        return {
            "image": image,                        # (H, W, 3)
            "world_pc_valid": world_pc_valid,      # (N, 4)
            "cam": cam,                             # ((3, 3), (4, 4))
            "dm": dm,                               # (H, W)
            "instance": os.path.split(self._rgb_path(sequence_path, frame_index))[1],  # string
        }

    # ------------------------- multi-frame api -------------------------

    def get_frame_infos(self, sequence_path: str, idxs: list[int]):
        """
        ultra-optimized multi-frame fetch from a single sequence.
        minimizes disk seeks, batches io with threads, and vectorizes ann access.

        returns: list[dict] in the same order as idxs
        """
        if not idxs:
            return []

        # sort indices to promote sequential disk access, track inverse permutation
        idxs_np = np.asarray(idxs, dtype=np.int32)                    # (F,)
        order = np.argsort(idxs_np)                                   # (F,)
        rev = np.empty_like(order)
        rev[order] = np.arange(order.size)
        idxs_sorted = idxs_np[order]

        # load annotations once (mmap to reduce peak rss)
        ann = np.load(f"{sequence_path}/anno.npz", allow_pickle=True, mmap_mode="r")
        intr_all = ann["intrinsics"]                                  # (T, 3, 3)
        extr_all = ann["extrinsics"]                                  # (T, 4, 4)
        trajs_all = ann["trajs_3d"]                                   # (T, Ni, 3) variable Ni
        visibs_all = ann["visibs"] if "visibs" in ann.files else ann["valids"]  # (T, Ni)

        # parallel image & depth reads (io-bound)
        def _load_pair(fid: int):
            rgb = self._read_rgb_bgr_to_rgb(self._rgb_path(sequence_path, fid))      # (H, W, 3)
            dm = self._read_depth_to_meters(self._depth_path(sequence_path, fid))    # (H, W)
            return fid, rgb, dm

        max_workers = min(8, max(1, (os.cpu_count() or 4) // 2), len(idxs_sorted))
        rgb_sorted = [None] * len(idxs_sorted)
        dm_sorted = [None] * len(idxs_sorted)

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_load_pair, int(fid)) for fid in idxs_sorted]
                for fut in as_completed(futs):
                    fid, rgb, dm = fut.result()
                    j = int(np.searchsorted(idxs_sorted, fid))  # idxs_sorted is sorted
                    rgb_sorted[j] = rgb
                    dm_sorted[j] = dm
        else:
            for j, fid in enumerate(idxs_sorted):
                rgb, dm = self._read_rgb_bgr_to_rgb(self._rgb_path(sequence_path, int(fid))), \
                          self._read_depth_to_meters(self._depth_path(sequence_path, int(fid)))
                rgb_sorted[j] = rgb
                dm_sorted[j] = dm

        # unsort to original order
        rgbs = [rgb_sorted[k] for k in rev]                            # list[(H, W, 3)]
        dms = [dm_sorted[k] for k in rev]                              # list[(H, W)]

        # gather per-frame ann with vectorized indexing where possible
        intr_batch = intr_all[idxs_np]                                 # (F, 3, 3)
        extr_batch = extr_all[idxs_np]                                 # (F, 4, 4)

        # package outputs
        results = [None] * len(idxs)
        for j, fidx in enumerate(idxs_np):
            world_pc = trajs_all[int(fidx)]                            # (N, 3)
            validity = visibs_all[int(fidx)][..., None].astype(np.float32)  # (N, 1)
            world_pc_valid = np.concatenate([world_pc, validity], axis=1).astype(np.float32)  # (N, 4)

            cam = (intr_batch[j], extr_batch[j])                      # ((3,3), (4,4))

            img = rgbs[j]
            dm = dms[j]

            # optional crop per-frame (kept here to preserve correctness of intrinsics)
            if self.config.data.crop:
                out_res = (self.config.data.size, self.config.data.size)  # (W, H)
                img, dm, cam = self.crop_data(img, dm, cam, out_res)

            results[j] = {
                "image": img,                         # (H, W, 3)
                "world_pc_valid": world_pc_valid,     # (N, 4)
                "cam": cam,                            # ((3, 3), (4, 4))
                "dm": dm,                              # (H, W)
                "instance": os.path.split(self._rgb_path(sequence_path, int(fidx)))[1],  # string
            }

        return results

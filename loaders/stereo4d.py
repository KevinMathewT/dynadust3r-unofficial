"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .stereo_motion_base import StereoMotionBase
# removed unused stereo4d_utils import
import utils.geometry as geom


def load_dataset_npz(path):
    """
    Load npz via memory-map, returning only the fields needed for Stereo4D:
      - track_lengths
      - track_indices
      - track_coordinates
      - camera2world
    """
    z = np.load(path, mmap_mode="r")
    return {
        "track_lengths": z["track_lengths"],
        "track_indices": z["track_indices"],
        "track_coordinates": z["track_coordinates"],
        "camera2world": z["camera2world"],
    }


import os
import pandas as pd


class Stereo4D(StereoMotionBase):
    """
    Dataset for stereo-motion sequences (Stereo4D).

    Scans annotation files in parallel to build a list of valid sequences
    and their frame counts, then precomputes random frame triplets.
    """

    def __init__(self, config, valid=False):
        super().__init__(config)
        print("loading stereo4d dataset...")

        self.dataset_label = config.dataset.stereo4d.name
        self.dataset_location = config.dataset.stereo4d.path
        self.max_frame_window = config.dataset.stereo4d.max_frame_window

        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.lefteye_dir = os.path.join(config.dataset.stereo4d.lefteye_dir, split)
        self.anno_dir = os.path.join(config.dataset.stereo4d.path, split)
        self.hfov = config.dataset.stereo4d.hfov

        meta_csv = os.path.join(
            config.dataset.stereo4d.meta_dir, "stereo4d_id_to_time_and_fov_metadata.csv"
        )
        stats_csv = os.path.join(config.dataset.stereo4d.meta_dir, "stats.csv")
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
        limit = len(stats_df) # min(len(stats_df), config.data.len)
        stats_df = stats_df.sample(n=limit, random_state=config.seed)

        for _, r in stats_df.iterrows():
            vid, cid = r["ytid"], r["clipid"]
            ts = meta_df.loc[
                (meta_df.vid == vid) & (meta_df.clipid == cid), "timestamp"
            ]
            if ts.empty:
                continue
            seq = f"{vid}_{int(ts.values[0])}"
            mp4 = os.path.join(self.lefteye_dir, f"{seq}-left_rectified.mp4")
            if not os.path.exists(mp4):
                continue
            self.sequence_paths.append(seq)
            self.frame_counts.append(int(r["d_frame"]))

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames: {sum(self.frame_counts)}")
        self._compute_triplets()

    # ---------- helpers ----------
    def _K(self, width):
        """
        Compute the camera intrinsic matrix for a square image of given width.

        Args:
            width (int): image width in pixels

        Returns:
            np.ndarray: 3×3 intrinsic matrix
        """
        fx = width * 0.5 / np.tan(np.deg2rad(self.hfov) * 0.5)
        return np.array([[fx, 0, width / 2], [0, fx, width / 2], [0, 0, 1]], np.float32)

    def _pc_for_frame(self, data, frame_idx):
        """
        Extract sparse 3D point cloud for a given frame.

        Args:
            data (dict): output of load_dataset_npz
            frame_idx (int): index of the frame within the sequence

        Returns:
            np.ndarray: (N,4) array of [x,y,z,valid_flag]
        """
        lengths = data["track_lengths"]
        indices = data["track_indices"]
        coords = data["track_coordinates"]
        num_tracks = len(lengths)
        points = np.zeros((num_tracks, 3), np.float32)
        valid = np.zeros((num_tracks, 1), np.float32)
        ptr = 0
        for i, L in enumerate(lengths):
            ids = indices[ptr : ptr + L]
            crd = coords[ptr : ptr + L]
            ptr += L
            mask = ids == frame_idx
            if mask.any():
                points[i] = crd[mask][0]
                valid[i] = 1

        return np.hstack([points, valid])

    # ---------- main API ----------
    def get_frame_info(self, seq, idx):
        """
        Load image and point cloud for a specific sequence and frame.

        Args:
            seq (str): sequence identifier
            idx (int): frame index

        Returns:
            dict: {
                image: (H,W,3) RGB array,
                world_pc_valid: (N,4) point cloud + valid flag,
                cam: ((3,3), (4,4)) intrinsics + extrinsics,
                dm: None,
                instance: unique string "{seq}_{idx:05d}"
            }
        """
        data = load_dataset_npz(os.path.join(self.anno_dir, f"{seq}.npz"))
        video_path = os.path.join(self.lefteye_dir, f"{seq}-left_rectified.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("read fail")
        img = img[:, :, ::-1]  # BGR→RGB

        world_pc = self._pc_for_frame(data, idx)
        # print(f"_pc_for_frame | seq: {seq} | valid: {world_pc[..., -1].sum()} | frame: {idx}")
        extrinsics = geom.inv(data["camera2world"][idx])
        intrinsics = self._K(img.shape[1])
        cam = (intrinsics, extrinsics)

        # if self.config.data.crop:
        #     size = self.config.data.size
        #     img, _, cam = self.crop_data(img, None, cam, (size, size))
        # print(f"_pc_for_frame-| seq: {seq} | valid: {world_pc[..., -1].sum()} | frame: {idx}")

        return dict(
            image=img,
            world_pc_valid=world_pc,
            cam=cam,
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )


import hydra
from omegaconf import DictConfig, open_dict


# usage:
# python -m loaders.stereo4d dataset.stereo4d.split=test
# python -m loaders.stereo4d dataset.stereo4d.split=test dataset.stereo4d.max_frame_window=30
# /scratch/projects/fouheylab/km6748/stereo4d-data/lefteye-perspective/train/yAAaVncw5g0_152118785-left_rectified.mp4
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    import torch
    import loaders.utils.viz as viz

    dataset_tracks = Stereo4D(config)
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

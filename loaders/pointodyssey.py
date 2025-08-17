"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import os
import cv2
import numpy as np

from .stereo_motion_base import StereoMotionBase
import utils.geometry as geom

class PointOdyssey(StereoMotionBase):
    def __init__(self, config, valid=False):
        """
        PointOdyssey dataset implementation
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        print("loading pointodyssey dataset...")

        self.dataset_label = config.dataset.pointodyssey.name
        self.dataset_location = config.dataset.pointodyssey.path
        self.max_frame_window = config.dataset.pointodyssey.max_frame_window
        
        # dataset-specific parames
        if not valid:
            self.split = config.dataset.pointodyssey.train_split
        else:
            self.split = config.dataset.pointodyssey.valid_split
        
        # init sequence paths
        split_dir = os.path.join(self.dataset_location, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Dataset split directory not found: {split_dir}")
        
        for item in os.listdir(split_dir):
            seq_path = os.path.join(split_dir, item)
            if os.path.isdir(seq_path) and \
               os.path.exists(os.path.join(seq_path, "rgbs")) and \
               os.path.exists(os.path.join(seq_path, "depths")) and \
               os.path.exists(os.path.join(seq_path, "anno.npz")):
                self.sequence_paths.append(seq_path)
        
        # count frames in each sequence
        self.frame_counts = []
        for seq_path in self.sequence_paths:
            rgb_dir = os.path.join(seq_path, "rgbs")
            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            self.frame_counts.append(len(rgb_files))
        
        # compute triplets
        self._compute_triplets()

    def get_frame_info(self, sequence_path, frame_index):
        """
        Get information for a specific frame with consistent cropping
        
        Args:
            sequence_path: Path to the sequence directory
            frame_index: Frame index
            
        Returns:
            Dictionary containing frame information
        """
        image = cv2.imread(f"{sequence_path}/rgbs/rgb_{frame_index:05d}.jpg")[:, :, ::-1]  # (H, W, 3)
        dm_16bit = cv2.imread(
            f"{sequence_path}/depths/depth_{frame_index:05d}.png", cv2.IMREAD_ANYDEPTH
        )  # (H, W)
        dm = dm_16bit.astype(np.float32) / 65535.0 * 1000.0 # convert to float depths # (H, W)

        annotations = np.load(f"{sequence_path}/anno.npz", allow_pickle=True)
        intrinsics = annotations["intrinsics"][frame_index]  # (3, 3)
        extrinsics = annotations["extrinsics"][frame_index]  # (4, 4)
        cam = (intrinsics, extrinsics) # ((3,3), (4,4))
        world_pc = annotations["trajs_3d"][frame_index]  # (N, 3)
        validity = annotations["visibs"][frame_index][..., np.newaxis]  # (N, 1)
        # validity = annotations["valids"][frame_index][..., np.newaxis]  # (N, 1)
        world_pc_valid = np.concatenate([world_pc, validity], axis=1)  # (N, 4)
        

        if self.config.data.crop:
            output_resolution = (self.config.data.size, self.config.data.size)  # (W, H)
            image, dm, cam = self.crop_data(image, dm, cam, output_resolution) # apply cropping
        
        return {
            "image": image,  # (H, W, 3)
            "world_pc_valid": world_pc_valid,  # (N, 4) scaled by factor "scale"
            "cam": cam,  # ((3, 3), (4, 4) homogeneous)
            "dm": dm,  # convert to meters, (H, W)
            "instance": os.path.split(f"{sequence_path}/rgbs/rgb_{frame_index:05d}.jpg")[1] # string
        }


import hydra
from omegaconf import DictConfig, open_dict

# usage: 
# python -m loaders.pointodyssey dataset.pointodyssey.split=sample
# python -m loaders.pointodyssey dataset.pointodyssey.split=sample dataset.pointodyssey.max_frame_window=30
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    import torch
    import loaders.utils.viz as viz
    
    # create dataset instance with 3d tracks
    with open_dict(config):
        config.dataset.pointodyssey.pm_source = "dm"
    
    dataset_tracks = PointOdyssey(config)
    print(f"total triplets: {len(dataset_tracks)}")
    
    for sample_idx in range(20, 30):
        sample_tracks = dataset_tracks[sample_idx]

        # print basic info about the triplet
        print(f"triplet frames: idxs={sample_tracks['idxs']}")
        print("------------------------")
        for k, v in sample_tracks.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size(), v.dtype)
            else:
                print(k)

        # prepare point maps and images for visualization
        track_pms = np.array(
            [sample_tracks["left_pm"], sample_tracks["mid_pm"], sample_tracks["right_pm"]]
        )

        c1 = sample_tracks["cam"]   # or sample_tracks["cam"]
        c2 = sample_tracks["cam_mid"]
        c3 = sample_tracks["cam_right"]

        img_l = sample_tracks["left_image"].permute(1,2,0).numpy()
        img_m = sample_tracks["mid_image"].permute(1,2,0).numpy()
        img_r = sample_tracks["right_image"].permute(1,2,0).numpy()

        images = [
            img_l,
            geom.recolor(track_pms[1], c1, c2, img_m),
            geom.recolor(track_pms[2], c1, c3, img_r),
        ]

        viz.visualize_image(
            sample_tracks["left_image"].permute(1, 2, 0).numpy(), name="left_image"
        )
        # viz.visualize_dm(sample_tracks['left_dm'], name="left_dm")
        viz.visualize_image(
            sample_tracks["mid_image"].permute(1, 2, 0).numpy(), name="mid_image"
        )
        # viz.visualize_dm(sample_tracks['mid_dm'], name="mid_dm")
        viz.visualize_image(
            sample_tracks["right_image"].permute(1, 2, 0).numpy(), name="right_image"
        )
        # viz.visualize_dm(sample_tracks['right_dm'], name="right_dm")

        viz.visualize_pm(
            track_pms[0], image=images[0], cam=sample_tracks["cam"], name="left_pm"
        )
        viz.visualize_pm(track_pms[1], image=images[1], cam=sample_tracks["cam"], name="mid_pm")
        viz.visualize_pm(
            track_pms[2], image=images[2], cam=sample_tracks["cam"], name="right_pm"
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
            name="tracksl",
        )

        viz.visualize_sequence_from_pms(
            np.array([track_pms[2], track_pms[1]]),  # right and mid for right_to_mid
            right_to_mid_map,
            [images[2], images[1]],
            name="tracksr",
        )


if __name__ == "__main__":
    main()

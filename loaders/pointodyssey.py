"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import os
import cv2
import numpy as np
from .stereo_motion_base import StereoMotionBase

class PointOdyssey(StereoMotionBase):
    def __init__(self, config, valid=False):
        """
        PointOdyssey dataset implementation
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        print("loading pointodyssey dataset...")
        
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
        Get information for a specific frame
        
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
        dm = dm_16bit.astype(np.float32) / 65535.0 * 1000.0

        annotations = np.load(f"{sequence_path}/anno.npz", allow_pickle=True)
        intrinsics = annotations["intrinsics"][frame_index]  # (3, 3)
        extrinsics = annotations["extrinsics"][frame_index]  # (4, 4) homogeneous
        cam = (intrinsics, extrinsics)
        world_pc = annotations["trajs_3d"][frame_index]  # (N, 3)
        validity = annotations["visibs"][frame_index][..., np.newaxis]  # (N, 1)
        # validity = annotations["valids"][frame_index][..., np.newaxis]  # (N, 1)
        world_pc_valid = np.concatenate([world_pc, validity], axis=1)  # (N, 4)

        return {
            "image": image,  # (H, W, 3)
            "world_pc_valid": world_pc_valid,  # (N, 4) scaled by factor "scale"
            "cam": cam,  # ((3, 3), (4, 4) homogeneous)
            "dm": dm,  # convert to meters, (H, W)
        }


import hydra
from omegaconf import DictConfig, open_dict

# usage: 
# python -m loaders.pointodyssey dataset.pointodyssey.split=sample
# python -m loaders.pointodyssey dataset.pointodyssey.split=sample dataset.pointodyssey.max_frame_window=30
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    import loaders.utils.viz as viz
    
    # create dataset instance with 3d tracks
    with open_dict(config):
        config.dataset.pointodyssey.pm_source = "3d_tracks"
    
    dataset_tracks = PointOdyssey(config)
    print(f"total triplets: {len(dataset_tracks)}")
    
    # get a sample triplet
    sample_idx = 31000
    sample_tracks = dataset_tracks[sample_idx]
    
    # print basic info about the triplet
    print(f"triplet frames: idxs={sample_tracks['idxs']}")
    
    # prepare point maps and images for visualization
    track_pms = np.array([
        sample_tracks['left_pm'],
        sample_tracks['mid_pm'],
        sample_tracks['right_pm']
    ])
    
    images = [
        sample_tracks['left_image'],
        sample_tracks['mid_image'],
        sample_tracks['right_image']
    ]
    
    # prepare motion maps
    left_to_mid = sample_tracks['left_to_mid_motion']
    right_to_mid = sample_tracks['right_to_mid_motion']
    
    # create single-step motion maps for each visualization
    h, w, _ = sample_tracks['left_pm'].shape
    
    # 1. tracks + left_to_mid
    left_to_mid_map = np.zeros((1, h, w, 4), dtype=np.float32)
    left_to_mid_map[0] = left_to_mid
    
    # 2. tracks + right_to_mid
    right_to_mid_map = np.zeros((1, h, w, 4), dtype=np.float32)
    right_to_mid_map[0] = right_to_mid
    
    # create dataset instance with depth maps
    with open_dict(config):
        config.dataset.pointodyssey.pm_source = "dm"
    
    dataset_dm = PointOdyssey(config)
    sample_dm = dataset_dm[sample_idx]
    
    # prepare point maps for depth-based visualization
    dm_pms = np.array([
        sample_dm['left_pm'],
        sample_dm['mid_pm'],
        sample_dm['right_pm']
    ])
    
    # visualize all 4 combinations
    print("visualizing 3d tracks + left to mid motion...")
    viz.visualize_sequence_from_pms(
        track_pms[:2],  # just left and mid for left_to_mid
        left_to_mid_map, 
        images[:2], 
        name="tracksl"
    )
    
    print("visualizing 3d tracks + right to mid motion...")
    viz.visualize_sequence_from_pms(
        np.array([track_pms[2], track_pms[1]]),  # right and mid for right_to_mid
        right_to_mid_map, 
        [images[2], images[1]], 
        name="tracksr"
    )
    
    print("visualizing depth maps + left to mid motion...")
    viz.visualize_sequence_from_pms(
        dm_pms[:2],  # just left and mid for left_to_mid
        left_to_mid_map, 
        images[:2], 
        name="dml"
    )
    
    print("visualizing depth maps + right to mid motion...")
    viz.visualize_sequence_from_pms(
        np.array([dm_pms[2], dm_pms[1]]),  # right and mid for right_to_mid
        right_to_mid_map, 
        [images[2], images[1]], 
        name="dmr"
    )

    # seq_path = "data/pointodyssey/sample/r4_new_f"
    # frame_info = dataset.get_frame_info(seq_path, 0)

    # for k, v in frame_info.items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v[0].shape, v[1].shape)

    # # viz.visualize_cam_movement_in_world(dataset, seq_path, num_frames=10)

    # world_pc_valid = frame_info["world_pc_valid"]
    # dm = frame_info["dm"]
    # cam = frame_info["cam"]
    # world_pc = np.asarray(world_pc_valid[:, :3])
    # cam_pc = geo.world_pc_to_cam_pc(world_pc, cam)
    # cam_dm_pc = geo.dm_to_cam_pc(dm, cam)
    # cam_pm = geo.cam_pc_to_cam_pm(cam_pc, cam, dm.shape)
    # cam_pm2 = geo.cam_pc_to_cam_pm(cam_dm_pc, cam, dm.shape)

    # print(f"world_pc_valid shape: {world_pc_valid.shape}")
    # print(f"world_pc_valid sample: {world_pc_valid[:5]}")
    # print(f"Depth map min/max: {dm.min()}, {dm.max()}")
    # print(f"Camera PC min/max: {cam_pc.min()}, {cam_pc.max()}")
    # print(f"Camera PC sample: {cam_pc[:5]}")
    # print(f"Camera DM PC min/max: {cam_dm_pc.min()}, {cam_dm_pc.max()}")
    # print(f"Camera DM PC sample: {cam_dm_pc[:5]}")

    # print("Min depth:", np.min(dm))
    # print("Max depth:", np.max(dm))
    # print("Number of (0,0,0) points:", np.sum((cam_pc == [0, 0, 0]).all(axis=1)))
    # print("Number of (0,0,0) points:", np.sum((cam_dm_pc == [0, 0, 0]).all(axis=1)))

    # print("Visualizing Point Cloud...")
    # viz.visualize_pc(
    #     cam_pc,
    #     image=frame_info["image"],
    #     cam=cam,
    #     valid=True,
    #     pc_in_cam_coords=True,
    # )
    # print("Visualizing Point Cloud Done.")

    # print("Visualizing Point Cloud...")
    # viz.visualize_pc(
    #     cam_dm_pc,
    #     image=frame_info["image"],
    #     cam=cam,
    #     valid=True,
    #     pc_in_cam_coords=True,
    #     # name="cam_dm_pc",
    # )
    # print("Visualizing Point Cloud Done.")

    # print("Visualizing Point Map...")
    # viz.visualize_pm(
    #     cam_pm,
    #     image=frame_info["image"],
    #     cam=cam,
    #     valid=True,
    #     pc_in_cam_coords=True,
    # )


    # print("Visualizing Point Map...")
    # viz.visualize_pm(
    #     cam_pm2,
    #     image=frame_info["image"],
    #     cam=cam,
    #     valid=True,
    #     pc_in_cam_coords=True,
    #     # name="cam_dm_pc",
    # )

    # t = 50

    # frame_infos = [dataset.get_frame_info(seq_path, i) for i in range(t)]
    # cams = [f["cam"] for f in frame_infos]
    # images = [f["image"] for f in frame_infos]

    # world_pcs = [f["world_pc_valid"][:, :3] for f in frame_infos]
    # valid_flags = [f["world_pc_valid"][..., 3:4] for f in frame_infos]

    # cam_pcs = [geo.world_pc_to_cam_pc(world_pcs[i], cams[0]) for i in range(t)]
    # cam_pc_valids = [np.concatenate([cam_pcs[i], valid_flags[i]], axis=1) for i in range(t)]

    # print(f"valid in cam_pc: {[int(cpv[:, -1].sum()) for cpv in cam_pc_valids]}")

    # cam_pms = [geo.cam_pc_to_cam_pm(cam_pc_valids[i], (cams[i][0], None), frame_infos[i]["dm"].shape, valid=True) for i in range(t)]
    # cam_dm_pms = [geo.dm_to_cam_pm(frame_infos[i]["dm"], cams[i]) for i in range(t)]
    # print(f"valid in cam_pm: {[cp[..., -1].sum() for cp in cam_pms]}")

    # motion_map = geo.get_motion_map_from_cam_pc(cam_pc_valids, cams[0][0], frame_infos[0]["dm"].shape)

    # print("--------------------------- total points -> valid points -> motion valid points")
    # for ti in range(motion_map.shape[0]):
    #     print(f"valid in motion_map at t={ti}: {frame_infos[ti]['world_pc_valid'].shape[0]} + {frame_infos[ti + 1]['world_pc_valid'].shape[0]} -> {int(cam_pc_valids[ti][:, -1].sum())} + {int(cam_pc_valids[ti + 1][:, -1].sum())} -> {int(motion_map[ti, ..., -1].sum())}")

    # viz.visualize_sequence_from_pms(np.asarray(cam_pms), motion_map, images, name="cam_pms_motion_map")
    # viz.visualize_sequence_from_pms(np.asarray(cam_dm_pms), motion_map, images, name="cam_dm_pms_motion_map")


if __name__ == "__main__":
    main()

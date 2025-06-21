import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import loaders.utils.geometry as geo


class StereoMotionBase(Dataset):
    def __init__(self, config, valid: bool = False):
        """
        Base class for stereo motion datasets that handles loading and processing of
        multi-view time series data for stereo and motion analysis.

        This class manages frame triplets (left, middle, right) across sequences, handles
        dataset indexing, and provides the infrastructure for 3D point cloud processing
        between camera views and timestamps. Intended to be subclassed for specific
        dataset implementations.

        Args:
            config: Configuration object containing dataset parameters including:
                - dataset.<dataset>.name: Name identifier for the dataset
                - dataset.<dataset>.path: Base directory path to dataset files
                - dataset.<dataset>.max_frame_window: Maximum temporal window size for frame triplets
            valid: Whether this is a validation dataset (affects triplet count)
        """
        # initialize common parameters
        self.config = config
        self.is_valid = valid  # Store whether this is validation dataset

        self.sequence_paths = []

        # populated by child classes
        self.frame_counts = []

    def _compute_triplets(self):
        """
        Compute a list of triplets through uniform sampling:
        1. Uniformly sample a sequence (each sequence has equal probability)
        2. Uniformly sample a left frame (ensuring room for right frame at min distance 2)
        3. Sample right frame between [left+2, left+max_frame_window]
        4. Sample mid frame between left and right

        The triplets are stored as (seq_idx, left, mid, right) tuples.
        """
        import numpy as np

        self.triplets = []  # initialize triplets list
        
        # Use appropriate target based on whether this is validation or training
        target_triplets = self.config.data.valid_len if self.is_valid else self.config.data.len
        
        while (
            len(self.triplets) < target_triplets
        ):  # generate triplets through uniform sampling
            seq_idx = np.random.randint(0, len(self.sequence_paths))  # sample sequence
            frame_count = self.frame_counts[seq_idx]

            if frame_count < 3:  # skip if too short
                continue

            max_left = frame_count - 3  # sample left frame
            if max_left < 0:
                continue
            left_frame = np.random.randint(0, max_left + 1)
            min_right = left_frame + 2  # sample right frame
            max_right = min(left_frame + self.max_frame_window, frame_count - 1)
            right_frame = np.random.randint(min_right, max_right + 1)
            mid_frame = np.random.randint(
                left_frame + 1, right_frame
            )  # sample mid frame
            self.triplets.append(
                (seq_idx, left_frame, mid_frame, right_frame)
            )  # add to list

        if len(self.triplets) > target_triplets:  # trim to exact size if needed
            self.triplets = self.triplets[:target_triplets]

        self.num_triplets = target_triplets  # set final size

    def __len__(self):
        """
        Returns the total number of triplets in the dataset.

        Returns:
            int: Total number of triplets in the dataset
        """
        return self.num_triplets

    def __getitem__(self, index):
        """
        Get a triplet of frames from the dataset based on the global triplet index.

        Args:
            index (int): Index into the precomputed triplets list

        Returns:
            dict: Dictionary containing the triplet data with PyTorch tensors:
                - left_pm, mid_pm, right_pm: Point maps for each frame (H, W, 4)
                - left_to_mid_motion, right_to_mid_motion: Motion maps between frames (H, W, 4)
                - left_image, mid_image, right_image: RGB images for each frame (3, H, W)
                - idxs: Frame indices tensor of shape (3,)
                - mid_tq: Normalized temporal position of mid frame between left and right (scalar tensor)
                - sequence_idx: Index of the sequence this triplet belongs to (scalar tensor)
                - left_instance, right_instance: Identifiers for left and right frames (string)
        """

        while True:
            # get triplet info
            seq_idx, left_frame, mid_frame, right_frame = self.triplets[index]
            sequence_path = self.sequence_paths[seq_idx]

            try:
                # fetch frame info
                left_info = self.get_frame_info(sequence_path, left_frame)
                mid_info = self.get_frame_info(sequence_path, mid_frame)
                right_info = self.get_frame_info(sequence_path, right_frame)
                
                # Assert that all frames have tracks
                assert len(left_info["world_pc_valid"]) > 0, f"left frame {sequence_path}:{left_frame} has no tracks"
                assert len(mid_info["world_pc_valid"]) > 0, f"mid frame {sequence_path}:{mid_frame} has no tracks"  
                assert len(right_info["world_pc_valid"]) > 0, f"right frame {sequence_path}:{right_frame} has no tracks"
                
                break
            except Exception as e:
                print(f"error loading triplet: {seq_idx}:{sequence_path}, {left_frame}, {mid_frame}, {right_frame}")
                print(f"error: {e}")
                index = np.random.randint(0, len(self.triplets))

        reference_cam = left_info["cam"]  # Use left camera as reference
        
        # Extract world points and validity
        left_world_pc = left_info["world_pc_valid"][:, :3]
        mid_world_pc = mid_info["world_pc_valid"][:, :3]
        right_world_pc = right_info["world_pc_valid"][:, :3]
        
        left_valid = left_info["world_pc_valid"][:, 3:4]
        mid_valid = mid_info["world_pc_valid"][:, 3:4]
        right_valid = right_info["world_pc_valid"][:, 3:4]

        # Get dataset config
        dataset_config = getattr(self.config.dataset, self.config.data.loader.lower(), None)
        pm_source = getattr(dataset_config, "pm_source", "") if dataset_config else ""
        
        if pm_source == "3d_tracks":
            # Create point maps with proper coordinate transforms
            left_pm = geo.create_pm_in_ref_frame(
                left_world_pc, left_valid, left_info["cam"], reference_cam, 
                left_info["image"].shape[:2], pm_source="3d_tracks"
            )
            
            mid_pm = geo.create_pm_in_ref_frame(
                mid_world_pc, mid_valid, mid_info["cam"], reference_cam, 
                mid_info["image"].shape[:2], pm_source="3d_tracks"
            )
            
            right_pm = geo.create_pm_in_ref_frame(
                right_world_pc, right_valid, right_info["cam"], reference_cam, 
                right_info["image"].shape[:2], pm_source="3d_tracks"
            )
            
        elif pm_source == "dm":
            # Create point maps from depth maps
            left_pm = geo.dm_to_cam_pm(left_info["dm"], reference_cam)
            
            mid_pm = geo.create_pm_from_dm_in_ref_frame(
                mid_info["dm"], mid_info["cam"], reference_cam
            )
            
            right_pm = geo.create_pm_from_dm_in_ref_frame(
                right_info["dm"], right_info["cam"], reference_cam
            )
            
        else:
            raise NotImplementedError("point map source not implemented")

        # ---------------------------------------------------------------
        # Compute motion maps: left→mid  and  right→mid
        # ---------------------------------------------------------------
        world_pc_valid_list = [
            left_info ["world_pc_valid"],   # (N,4) left
            right_info["world_pc_valid"],   # (N,4) right
            mid_info  ["world_pc_valid"],   # (N,4) mid   ← target
        ]
        cam_list = [
            left_info ["cam"],
            right_info["cam"],
            mid_info  ["cam"],              # ← target
        ]

        motion_maps = geo.get_motion_map_from_world_pc(
            world_pc_valid_list,
            cam_list,
            left_info["image"].shape[:2],   # (H,W) – any is fine; function only uses per-frame intrinsics
        )
        left_to_mid_motion  = motion_maps[0]   # stored on left grid
        right_to_mid_motion = motion_maps[1]   # stored on right grid

        # Process images
        m = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        s = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        left_image = (left_info["image"].astype(np.float32) / 255 - m) / s
        mid_image = (mid_info["image"].astype(np.float32) / 255 - m) / s
        right_image = (right_info["image"].astype(np.float32) / 255 - m) / s

        return {
            "left_pm": torch.from_numpy(left_pm.copy()).float(),  # (H, W, 4)
            "mid_pm": torch.from_numpy(mid_pm.copy()).float(),  # (H, W, 4)
            "right_pm": torch.from_numpy(right_pm.copy()).float(),  # (H, W, 4)
            "left_to_mid_motion": torch.from_numpy(left_to_mid_motion.copy()).float(),  # (H, W, 4)
            "right_to_mid_motion": torch.from_numpy(right_to_mid_motion.copy()).float(),  # (H, W, 4)
            "left_image": torch.from_numpy(left_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            "mid_image": torch.from_numpy(mid_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            "right_image": torch.from_numpy(right_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            "idxs": torch.tensor((left_frame, mid_frame, right_frame)).float(),  # tensor of shape (3,)
            "mid_tq": torch.tensor((mid_frame - left_frame) / (right_frame - left_frame)).float(),  # scalar tensor
            "sequence_idx": torch.tensor(seq_idx).float(),  # scalar tensor
            "left_instance": left_info.get("instance", None),  # string
            "right_instance": right_info.get("instance", None),  # string
            "cam": left_info["cam"],  # ((3,3), (4,4))
            "cam_mid": mid_info["cam"],  # ((3,3), (4,4))
            "cam_right": right_info["cam"],  # ((3,3), (4,4))
        }


    def crop_data(self, image, dm, cam, output_resolution):
        """
        Apply consistent cropping and resizing to image, depth map, and camera parameters.

        Args:
            image: RGB image array (H, W, 3)
            dm: Depth map array (H, W)
            cam: Camera parameters tuple ((3,3) intrinsics, (4,4) extrinsics)
            output_resolution: Target output resolution (width, height)

        Returns:
            tuple: (cropped_image, cropped_dm, adjusted_cam)
        """
        import loaders.utils.geometry as geo

        intrinsics, extrinsics = cam  # ((3,3), (4,4))

        cropped_image, cropped_dm, adjusted_intrinsics = geo.crop_resize_if_necessary(
            image, dm, intrinsics, output_resolution
        )  # apply transformations

        if isinstance(cropped_image, Image.Image):
            cropped_image = np.array(cropped_image)  # (H, W, 3)

        adjusted_cam = (adjusted_intrinsics, extrinsics)  # ((3,3), (4,4))

        return cropped_image, cropped_dm, adjusted_cam

    def get_frame_info(self, sequence_path, frame_index):
        """
        Abstract method to retrieve comprehensive information for a specific frame.

        This method must be implemented by child classes to handle dataset-specific
        frame loading and processing. It should load all necessary data for a frame
        including images, depth maps, camera parameters, and point clouds.

        Args:
            sequence_path (str): Path to the sequence directory containing frame data
            frame_index (int): Index of the frame to retrieve within the sequence

        Returns:
            dict: Dictionary containing frame information with at minimum these keys:
                - image: RGB image array (H, W, 3)
                - dm: Depth map array (H, W)
                - cam: Camera parameters tuple ((3,3) intrinsic matrix, (4,4) extrinsic matrix)
                - world_pc_valid: World point cloud with validity flags (N, 4)
                  where the first 3 columns are XYZ coordinates and the 4th is a validity flag

        Raises:
            NotImplementedError: This method must be overridden by child classes
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

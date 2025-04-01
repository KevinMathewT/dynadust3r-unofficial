import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import loaders.utils.geometry as geo

class StereoMotionBase(Dataset):
    def __init__(self, config):
        """
        Base class for stereo motion datasets that handles loading and processing of
        multi-view time series data for stereo and motion analysis.
        
        This class manages frame triplets (left, middle, right) across sequences, handles 
        dataset indexing, and provides the infrastructure for 3D point cloud processing 
        between camera views and timestamps. Intended to be subclassed for specific 
        dataset implementations.
        
        Args:
            config: Configuration object containing dataset parameters including:
                - dataset.pointodyssey.name: Name identifier for the dataset
                - dataset.pointodyssey.path: Base directory path to dataset files
                - dataset.pointodyssey.max_frame_window: Maximum temporal window size for frame triplets
        """
        # initialize common parameters
        self.config = config
        self.dataset_label = config.dataset.pointodyssey.name
        self.dataset_location = config.dataset.pointodyssey.path
        self.max_frame_window = config.dataset.pointodyssey.max_frame_window
        
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
        
        # initialize triplets list
        self.triplets = []
        
        # get target dataset size from config
        target_triplets = self.config.data.size
        
        # generate triplets through uniform sampling
        while len(self.triplets) < target_triplets:
            # sample sequence
            seq_idx = np.random.randint(0, len(self.sequence_paths))
            frame_count = self.frame_counts[seq_idx]
            
            # skip if too short
            if frame_count < 3:
                continue
            
            # sample left frame
            max_left = frame_count - 3
            if max_left < 0:
                continue
            left_frame = np.random.randint(0, max_left + 1)
            
            # sample right frame
            min_right = left_frame + 2
            max_right = min(left_frame + self.max_frame_window, frame_count - 1)
            right_frame = np.random.randint(min_right, max_right + 1)
            
            # sample mid frame
            mid_frame = np.random.randint(left_frame + 1, right_frame)
            
            # add to list
            self.triplets.append((seq_idx, left_frame, mid_frame, right_frame))
        
        # trim to exact size if needed
        if len(self.triplets) > target_triplets:
            self.triplets = self.triplets[:target_triplets]
        
        # set final size
        self.num_triplets = target_triplets
        
    
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
        """
        # get triplet info
        seq_idx, left_frame, mid_frame, right_frame = self.triplets[index]
        sequence_path = self.sequence_paths[seq_idx]
        
        # fetch frame info
        left_info = self.get_frame_info(sequence_path, left_frame)  # dict
        mid_info = self.get_frame_info(sequence_path, mid_frame)  # dict
        right_info = self.get_frame_info(sequence_path, right_frame)  # dict
        
        reference_cam = left_info["cam"]  # ((3, 3), (4, 4))
        image_shape = left_info["dm"].shape  # (H, W)
        
        left_world_pc = left_info["world_pc_valid"][:, :3]  # (N_left, 3)
        mid_world_pc = mid_info["world_pc_valid"][:, :3]  # (N_mid, 3)
        right_world_pc = right_info["world_pc_valid"][:, :3]  # (N_right, 3)
        
        left_valid = left_info["world_pc_valid"][:, 3:4]  # (N_left, 1)
        mid_valid = mid_info["world_pc_valid"][:, 3:4]  # (N_mid, 1)
        right_valid = right_info["world_pc_valid"][:, 3:4]  # (N_right, 1)
        
        # convert points to camera coordinates
        left_cam_pc = geo.world_pc_to_cam_pc(left_world_pc, reference_cam)  # (N_left, 3)
        mid_cam_pc = geo.world_pc_to_cam_pc(mid_world_pc, reference_cam)  # (N_mid, 3)
        right_cam_pc = geo.world_pc_to_cam_pc(right_world_pc, reference_cam)  # (N_right, 3)
        
        left_cam_pc_valid = np.concatenate([left_cam_pc, left_valid], axis=1)  # (N_left, 4)
        mid_cam_pc_valid = np.concatenate([mid_cam_pc, mid_valid], axis=1)  # (N_mid, 4)
        right_cam_pc_valid = np.concatenate([right_cam_pc, right_valid], axis=1)  # (N_right, 4)
        
        image_shape = left_info["dm"].shape  # (H, W)
        
        # get dataset config
        dataset_config = getattr(self.config.dataset, self.dataset_label.lower(), None)
        if dataset_config is not None and getattr(dataset_config, "pm_source", "") == "3d_tracks":
            left_pm = geo.cam_pc_to_cam_pm(left_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            mid_pm = geo.cam_pc_to_cam_pm(mid_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            right_pm = geo.cam_pc_to_cam_pm(right_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
        
        elif dataset_config is not None and getattr(dataset_config, "pm_source", "") == "dm":
            left_pm = geo.dm_to_cam_pm(left_info["dm"], reference_cam)  # (H, W, 4)
            
            # process mid frame
            mid_dm_pc = geo.dm_to_cam_pc(mid_info["dm"], mid_info["cam"])  # (N_mid_dense, 3)
            mid_dm_world_pc = geo.cam_pc_to_world_pc(mid_dm_pc, mid_info["cam"])  # (N_mid_dense, 3)
            mid_dm_left_pc = geo.world_pc_to_cam_pc(mid_dm_world_pc, reference_cam)  # (N_mid_dense, 3)
            mid_pm = geo.cam_pc_to_cam_pm(mid_dm_left_pc, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            
            # process right frame
            right_dm_pc = geo.dm_to_cam_pc(right_info["dm"], right_info["cam"])  # (N_right_dense, 3)
            right_dm_world_pc = geo.cam_pc_to_world_pc(right_dm_pc, right_info["cam"])  # (N_right_dense, 3)
            right_dm_left_pc = geo.world_pc_to_cam_pc(right_dm_world_pc, reference_cam)  # (N_right_dense, 3)
            right_pm = geo.cam_pc_to_cam_pm(right_dm_left_pc, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
        else:
            raise NotImplementedError("point map source not implemented")
            
        # compute motion maps
        left_to_mid_motion = geo.get_motion_map_from_cam_pc(
            [left_cam_pc_valid, mid_cam_pc_valid], 
            reference_cam[0], 
            image_shape
        )[0]  # (H, W, 4)
        
        right_to_mid_motion = geo.get_motion_map_from_cam_pc(
            [right_cam_pc_valid, mid_cam_pc_valid], 
            reference_cam[0], 
            image_shape
        )[0]  # (H, W, 4)
        
        return {
            "left_pm": torch.from_numpy(left_pm.copy()),  # (H, W, 4)
            "mid_pm": torch.from_numpy(mid_pm.copy()),  # (H, W, 4)
            "right_pm": torch.from_numpy(right_pm.copy()),  # (H, W, 4)
            "left_to_mid_motion": torch.from_numpy(left_to_mid_motion.copy()),  # (H, W, 4)
            "right_to_mid_motion": torch.from_numpy(right_to_mid_motion.copy()),  # (H, W, 4)
            "left_image": torch.from_numpy(left_info["image"].copy()).permute(2, 0, 1),  # (3, H, W)
            "mid_image": torch.from_numpy(mid_info["image"].copy()).permute(2, 0, 1),  # (3, H, W)
            "right_image": torch.from_numpy(right_info["image"].copy()).permute(2, 0, 1),  # (3, H, W)
            "idxs": torch.tensor((left_frame, mid_frame, right_frame)),  # tensor of shape (3,)
            "mid_tq": torch.tensor((mid_frame - left_frame) / (right_frame - left_frame)),  # scalar tensor
            "sequence_idx": torch.tensor(seq_idx),  # scalar tensor
        }
    
    
    
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
        raise NotImplementedError("This method should be implemented in the child class")
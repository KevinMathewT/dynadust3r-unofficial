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
        # Initialize common parameters
        self.config = config
        self.dataset_label = config.dataset.pointodyssey.name
        self.dataset_location = config.dataset.pointodyssey.path
        self.max_frame_window = config.dataset.pointodyssey.max_frame_window
        
        self.sequence_paths = []
        
        # These will be populated by child classes
        self.frame_counts = []
        self.triplets_per_seq = []
        self.cum_triplets = []
    
    def _compute_triplets(self):
        """
        Compute the number of triplets per sequence and cumulative triplet counts across sequences.
        
        For each starting position, we generate (max_frame_window-1) possible middle frames.
        Each sequence can have multiple starting positions based on its length, with 
        total triplets = (count - max_frame_window) * (max_frame_window-1) per sequence.
        
        This method should be called after populating self.frame_counts in the child class 
        implementation. The results are stored in self.triplets_per_seq and self.cum_triplets 
        for efficient indexing during dataset iteration.
        """
        # for each starting position, we have (max_frame_window-1) possible mid frames
        # total triplets = (count - max_frame_window) * (max_frame_window-1)
        self.triplets_per_seq = [max(0, count - self.max_frame_window) * (self.max_frame_window - 1) 
                                for count in self.frame_counts]  # [scalar]
        
        self.cum_triplets = np.cumsum([0] + self.triplets_per_seq)  # [scalar]
    
    def __len__(self):
        """
        Returns the total number of triplets in the dataset across all sequences.
        
        Returns:
            int: Total number of frame triplets available in the dataset.
                 Returns 0 if no triplets have been computed yet.
        """
        return self.cum_triplets[-1] if len(self.cum_triplets) > 0 else 0  # scalar
    
    def __getitem__(self, index):
        """
        Get a triplet of frames from the dataset based on a global triplet index.
        
        This method:
        1. Identifies the sequence and local index from the global index
        2. Calculates the specific frame indices for left, middle, and right frames
        3. Retrieves frame information for each position
        4. Computes point maps, transformations, and motion maps between frames
        5. Returns a comprehensive dictionary with aligned data across the triplet
        
        Args:
            index (int): Global triplet index across all sequences
            
        Returns:
            dict: Dictionary containing the triplet data with keys:
                - left_pm, mid_pm, right_pm: Point maps for each frame (H, W, 4)
                - left_to_mid_motion, right_to_mid_motion: Motion maps between frames (H, W, 4)
                - left_image, mid_image, right_image: RGB images for each frame (H, W, 3)
                - idxs: Frame indices tuple (left_frame, mid_frame, right_frame)
                - mid_tq: Normalized temporal position of mid frame between left and right
                - sequence_idx: Index of the sequence this triplet belongs to
        """
        seq_idx = np.searchsorted(self.cum_triplets[1:], index, side='right')  # scalar
        local_idx = index - self.cum_triplets[seq_idx]  # scalar
        sequence_path = self.sequence_paths[seq_idx]  # string
        start_pos = local_idx // (self.max_frame_window - 1)  # scalar
        mid_offset = (local_idx % (self.max_frame_window - 1)) + 1  # scalar
        left_frame = start_pos  # scalar
        right_frame = left_frame + self.max_frame_window  # scalar
        mid_frame = left_frame + mid_offset  # scalar
        
        # child class will implement get_frame_info
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
        
        # Convert all points to world coordinates first, then to mid-camera coordinates
        left_cam_pc = geo.world_pc_to_cam_pc(left_world_pc, reference_cam)  # (N_left, 3)
        mid_cam_pc = geo.world_pc_to_cam_pc(mid_world_pc, reference_cam)  # (N_mid, 3)
        right_cam_pc = geo.world_pc_to_cam_pc(right_world_pc, reference_cam)  # (N_right, 3)
        
        left_cam_pc_valid = np.concatenate([left_cam_pc, left_valid], axis=1)  # (N_left, 4)
        mid_cam_pc_valid = np.concatenate([mid_cam_pc, mid_valid], axis=1)  # (N_mid, 4)
        right_cam_pc_valid = np.concatenate([right_cam_pc, right_valid], axis=1)  # (N_right, 4)
        
        image_shape = left_info["dm"].shape  # (H, W)
        
        # Get the dataset config based on the dataset label
        dataset_config = getattr(self.config.dataset, self.dataset_label.lower(), None)
        if dataset_config is not None and getattr(dataset_config, "pm_source", "") == "3d_tracks":
            left_pm = geo.cam_pc_to_cam_pm(left_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            mid_pm = geo.cam_pc_to_cam_pm(mid_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            right_pm = geo.cam_pc_to_cam_pm(right_cam_pc_valid, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
        elif dataset_config is not None and getattr(dataset_config, "pm_source", "") == "dm":
            left_pm = geo.dm_to_cam_pm(left_info["dm"], reference_cam)  # (H, W, 4)
            
            mid_dm_pc = geo.dm_to_cam_pc(mid_info["dm"], mid_info["cam"])  # (N_mid_dense, 3)
            mid_dm_world_pc = geo.cam_pc_to_world_pc(mid_dm_pc, mid_info["cam"])  # (N_mid_dense, 3)
            mid_dm_left_pc = geo.world_pc_to_cam_pc(mid_dm_world_pc, reference_cam)  # (N_mid_dense, 3)
            mid_pm = geo.cam_pc_to_cam_pm(mid_dm_left_pc, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
            
            right_dm_pc = geo.dm_to_cam_pc(right_info["dm"], right_info["cam"])  # (N_right_dense, 3)
            right_dm_world_pc = geo.cam_pc_to_world_pc(right_dm_pc, right_info["cam"])  # (N_right_dense, 3)
            right_dm_left_pc = geo.world_pc_to_cam_pc(right_dm_world_pc, reference_cam)  # (N_right_dense, 3)
            right_pm = geo.cam_pc_to_cam_pm(right_dm_left_pc, (reference_cam[0], None), image_shape, valid=True)  # (H, W, 4)
        else:
            raise NotImplementedError("point map source not implemented")
            
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
            "left_pm": left_pm,  # (H, W, 4)
            "mid_pm": mid_pm,  # (H, W, 4)
            "right_pm": right_pm,  # (H, W, 4)
            "left_to_mid_motion": left_to_mid_motion,  # (H, W, 4)
            "right_to_mid_motion": right_to_mid_motion,  # (H, W, 4)
            "left_image": left_info["image"],  # (H, W, 3)
            "mid_image": mid_info["image"],  # (H, W, 3)
            "right_image": right_info["image"],  # (H, W, 3)
            "idxs": (left_frame, mid_frame, right_frame),  # tuple of 3 scalars
            "mid_tq": (mid_frame - left_frame) / (right_frame - left_frame),  # scalar
            "sequence_idx": seq_idx,  # scalar
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
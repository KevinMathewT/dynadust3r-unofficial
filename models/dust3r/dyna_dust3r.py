import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dust3r.model import AsymmetricCroCo3DStereo
from models.dust3r.utils.heads import head_factory
from models.dust3r.utils.misc import transpose_to_landscape
from models.dust3r.utils.heads.postprocess import postprocess
class DynaDUSt3R(AsymmetricCroCo3DStereo):
    """
    Extension of DUSt3R for dynamic scenes. Adds a motion head to predict 3D
    motion vectors between frames at a specified query time.
    """
    def __init__(self,
                output_mode='pts3d',
                head_type='dpt',
                depth_mode=('exp', -float('inf'), float('inf')),
                conf_mode=('exp', 1, float('inf')),
                motion_conf_mode=('exp', 1, float('inf')),
                freeze='none',
                landscape_only=True,
                patch_embed_cls='PatchEmbedDust3R',
                time_embedding_dim=128,
                **croco_kwargs):
        # Initialize parent class
        super().__init__(output_mode=output_mode,
                        head_type=head_type,
                        depth_mode=depth_mode,
                        conf_mode=conf_mode,
                        freeze=freeze,
                        landscape_only=landscape_only,
                        patch_embed_cls=patch_embed_cls,
                        **croco_kwargs)
        
        # Store additional configurations
        self.motion_conf_mode = motion_conf_mode
        self.time_embedding_dim = time_embedding_dim
        
        # Initialize time embedding
        self._init_time_embedding()
        
        # Initialize motion heads alongside the existing points heads
        self._set_motion_heads(output_mode, head_type, landscape_only)

    def _init_time_embedding(self):
        """Initialize the time embedding module."""
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_embedding_dim),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim)
        )

    def _set_motion_heads(self, output_mode, head_type, landscape_only):
        """Initialize motion prediction heads."""
        # Create motion heads with the same architecture as the point heads
        self.motion_downstream_head1 = MotionHead(self, head_type, output_mode, 
                                                bool(self.motion_conf_mode), 
                                                self.time_embedding_dim)
        self.motion_downstream_head2 = MotionHead(self, head_type, output_mode, 
                                                bool(self.motion_conf_mode), 
                                                self.time_embedding_dim)
        
        # Apply landscape transformation if needed
        self.motion_head1 = transpose_to_landscape(self.motion_downstream_head1, activate=landscape_only)
        self.motion_head2 = transpose_to_landscape(self.motion_downstream_head2, activate=landscape_only)

    def forward(self, view1, view2, t_query=None):
        """
        Forward pass that processes two views and optionally a query time.
        
        Args:
            view1: First view
            view2: Second view
            t_query: Query time in [0, 1], where 0 corresponds to view1's time
                    and 1 corresponds to view2's time. If None, only point prediction is done.
                    
        Returns:
            res1, res2: Results for each view with 3D points and motion vectors
        """
        # Encode the two images
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        # Get decoder outputs
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        
        # Predict 3D points using the parent class's downstream heads
        with torch.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        
        # Remap the output for view2 as in parent class
        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        
        # If t_query is provided, compute motion vectors
        if t_query is not None:
            # Compute time embedding
            B = dec1[0].shape[0]  # Batch size
            t_tensor = torch.full((B,), t_query, device=dec1[0].device)
            t_emb = self.time_embed(t_tensor)
            
            # Predict motion vectors using the motion heads
            with torch.amp.autocast(enabled=False):
                motion1 = self.motion_head1([tok.float() for tok in dec1], shape1, t_query, t_emb)
                motion2 = self.motion_head2([tok.float() for tok in dec2], shape2, t_query, t_emb)
            
            # Add motion results to the output
            res1['motion_vectors'] = motion1['pts3d']
            if 'confidence' in motion1:
                res1['motion_confidence'] = motion1['confidence']
                
            res2['motion_vectors_in_other_view'] = motion2['pts3d']
            if 'confidence' in motion2:
                res2['motion_confidence'] = motion2['confidence']
    
        return res1, res2
    
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class MotionHead(nn.Module):
    """
    Motion prediction head that integrates time embedding for dynamic scene prediction.
    Supports both linear and DPT head architectures.
    """
    def __init__(self, net, head_type, output_mode, has_conf, time_embedding_dim):
        super().__init__()
        self.net = net
        self.head_type = head_type
        self.time_embedding_dim = time_embedding_dim
        
        # Create the base head using the factory method
        if head_type == 'linear':
            self.base_head = head_factory(head_type, output_mode, net, has_conf=has_conf)
            # For linear head, create a simple projection for time embedding
            self.time_proj = nn.Sequential(
                nn.Linear(time_embedding_dim, net.dec_embed_dim),
                nn.SiLU(),
                nn.Linear(net.dec_embed_dim, net.dec_embed_dim)
            )
        else:  # 'dpt'
            # For DPT head, we need to create a similar head to the point head
            # but we'll modify it to integrate time information
            self.base_head = head_factory(head_type, output_mode, net, has_conf=has_conf)
            
            # Setup time projections for each DPT layer
            # Standard for DPT
            feature_dim = 256
            n_layers = 4
            
            # Create time projections for each layer
            self.time_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(time_embedding_dim, feature_dim),
                    nn.SiLU(),
                    nn.Linear(feature_dim, feature_dim)
                ) for _ in range(n_layers)
            ])

    def forward(self, decout, img_shape, t_query, t_emb):
        """
        Forward pass for motion prediction.
        
        Args:
            decout: Decoder outputs
            img_shape: Image shape
            t_query: Query time in [0, 1]
            t_emb: Time embedding tensor
            
        Returns:
            Motion vectors
        """
        if self.head_type == 'linear':
            # For linear head, simply integrate time with the last decoder output
            time_proj = self.time_proj(t_emb)
            time_integrated = decout[-1] + time_proj.unsqueeze(1)
            x_with_time = decout[:-1] + [time_integrated]
            
            # Forward with time-integrated features
            return self.base_head(x_with_time, img_shape)
        else:  # 'dpt'
            # For DPT head, we need to access the internal structure
            # and modify the layer processing to incorporate time
            
            # Access the DPT object inside the PixelwiseTaskWithDPT
            dpt = self.base_head.dpt
            
            # Store original layer_rn functions
            original_layer_rn = []
            for i in range(len(dpt.scratch.layer_rn)):
                original_layer_rn.append(dpt.scratch.layer_rn[i])
            
            # Create patched functions for each layer that add time embedding
            def make_time_layer(idx, original_fn, time_proj_fn):
                def time_layer(x):
                    # Project time embedding to the right dimension and shape
                    B = x.shape[0]
                    time_proj = time_proj_fn(t_emb).view(B, -1, 1, 1)
                    # Apply original function and add time embedding
                    return original_fn(x) + time_proj
                return time_layer
            
            # Replace layer_rn functions with time-aware versions
            for i in range(len(dpt.scratch.layer_rn)):
                dpt.scratch.layer_rn[i] = make_time_layer(
                    i, original_layer_rn[i], self.time_projections[i]
                )
            
            try:
                # Run the prediction with patched layers
                return self.base_head(decout, img_shape)
            finally:
                # Restore original functions
                for i in range(len(dpt.scratch.layer_rn)):
                    dpt.scratch.layer_rn[i] = original_layer_rn[i]
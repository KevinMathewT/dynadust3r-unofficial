import torch
import torch.nn as nn
import math


class TimePositionEmbedding(nn.Module):
    """
    Time-based position embedding using sinusoidal encoding.
    Encodes a scalar time value t ∈ [0, 1] into a high-dimensional vector
    using sinusoidal functions at different frequencies.
    """
    def __init__(self, embedding_dim=128, max_freq_log2=10):
        """
        Initialize time position embedding.
        
        Args:
            embedding_dim (int): Dimension of the time embedding. Must be even.
            max_freq_log2 (int): Maximum frequency (in log2 space)
        """
        super().__init__()
        
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        self.embedding_dim = embedding_dim
        self.max_freq_log2 = max_freq_log2
        
        # Create frequency bands: each pair of dimensions gets a frequency
        # These frequencies increase exponentially
        self.freq_bands = 2.0 ** torch.linspace(0, max_freq_log2, embedding_dim // 2)
        print(torch.linspace(0, max_freq_log2, embedding_dim // 2))
        print(self.freq_bands)

    def forward(self, t):
        """
        Compute sinusoidal position embedding for time t.
        
        Args:
            t (torch.Tensor): Time values of shape [batch_size, 1] in range [0, 1]
            
        Returns:
            torch.Tensor: Time embeddings of shape [batch_size, embedding_dim]
        """
        # Ensure t has the right shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Scale t by frequency bands: [batch_size, freq_bands]
        t_scaled = t * self.freq_bands.to(t.device)
        
        # Calculate sin and cos for each frequency
        # output shape: [batch_size, embedding_dim]
        embedding = torch.cat(
            [torch.sin(t_scaled), torch.cos(t_scaled)], 
            dim=-1
        )
        
        return embedding


class TimeEmbeddingMLP(nn.Module):
    """
    MLP for projecting time embeddings to feature dimensions.
    Architecture: Dense -> SiLU -> Dense
    """
    def __init__(self, input_dim=128, output_dim=256):
        """
        Initialize time embedding projection MLP.
        
        Args:
            input_dim (int): Dimension of input time embedding
            output_dim (int): Dimension of output features
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Project time embedding to feature dimension.
        
        Args:
            x (torch.Tensor): Time embedding of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Projected features of shape [batch_size, output_dim]
        """
        return self.mlp(x)


class TimeFeatureInjector(nn.Module):
    """
    Injects time embeddings into network features.
    Creates a separate MLP for each feature level.
    """
    def __init__(self, embedding_dim=128, feature_dims=[256, 256, 256, 256]):
        """
        Initialize time feature injector.
        
        Args:
            embedding_dim (int): Dimension of time embedding
            feature_dims (list): List of feature dimensions for each level
        """
        super().__init__()
        
        self.time_embed = TimePositionEmbedding(embedding_dim=embedding_dim)
        
        # Create a separate MLP for each feature level
        self.projectors = nn.ModuleList([
            TimeEmbeddingMLP(embedding_dim, dim) 
            for dim in feature_dims
        ])
    
    def forward(self, t, features):
        """
        Inject time embeddings into features.
        
        Args:
            t (torch.Tensor): Time values of shape [batch_size, 1] in range [0, 1]
            features (list): List of feature tensors
            
        Returns:
            list: Time-modulated features
        """
        # Get time embedding
        time_embed = self.time_embed(t)
        
        # Project and add to each feature level
        # Each projection is unsqueezed to match spatial dimensions
        time_features = []
        for idx, feature in enumerate(features):
            # Project time embedding to feature dimension
            proj = self.projectors[idx](time_embed)
            
            # Add spatial dimensions to match feature shape
            while proj.dim() < feature.dim():
                proj = proj.unsqueeze(-1)
                
            # Add time projection to feature
            time_features.append(feature + proj)
            
        return time_features
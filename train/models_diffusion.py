"""
Diffusion-based Projector model
Replaces the original projector network with a diffusion model
Maintains the same input/output format as the original framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DiffusionProjector(nn.Module):
    """
    Diffusion-based Projector network
    Uses a diffusion model to project query features onto nearest features + latent variables
    
    Input: query feature vector (nfeatures)
    Output: projected features (nfeatures) + projected latent variables (nlatent)
    
    The model follows the same interface as the original Projector:
    - Takes normalized query features as input
    - Outputs denormalized features + latent variables
    """
    
    def __init__(self, input_size, output_size, hidden_size=512, num_timesteps=1000):
        super(DiffusionProjector, self).__init__()
        
        self.input_size = input_size  # nfeatures
        self.output_size = output_size  # nfeatures + nlatent
        self.num_timesteps = num_timesteps
        
        # Time embedding
        time_dim = hidden_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_size // 4),
            nn.Linear(hidden_size // 4, time_dim),
            nn.ReLU()
        )
        
        # Main network - U-Net style architecture
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Middle block
        self.middle = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Decoder blocks
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, timestep=None):
        """
        Forward pass - compatible with original projector interface
        
        Args:
            x: input features (batch_size, input_size) - normalized query features
            timestep: diffusion timestep (batch_size,) - if None, uses random timestep during training
        
        Returns:
            output: (batch_size, output_size) - normalized features + latent (will be denormalized by caller)
            
        Note: For inference (compatibility with C++ framework), call with timestep=0
        """
        batch_size = x.shape[0]
        
        # If timestep is None (during training), sample random timesteps
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        else:
            timestep = timestep.long()
        
        # Time embedding
        t_emb = self.time_mlp(timestep.float())
        
        # Input projection
        h = self.input_proj(x)
        h = h + t_emb  # Add time embedding
        
        # Encoder
        h1 = self.encoder1(h)
        h = h + h1  # Residual connection
        
        h2 = self.encoder2(h)
        h = h + h2  # Residual connection
        
        # Middle
        h_mid = self.middle(h)
        h = h + h_mid
        
        # Decoder with skip connections
        h = torch.cat([h, h2], dim=-1)
        h3 = self.decoder1(h)
        h = h3
        
        h = torch.cat([h, h1], dim=-1)
        h4 = self.decoder2(h)
        h = h4
        
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def predict_noise(self, x_noisy, timestep):
        """
        Predict the noise added to the clean signal
        This is the standard diffusion model interface
        """
        return self.forward(x_noisy, timestep)
    
    def sample(self, x_query, num_steps=50, guidance_scale=1.0):
        """
        Sample from the diffusion model (DDIM sampling)
        
        Args:
            x_query: query features (batch_size, input_size) - normalized
            num_steps: number of sampling steps
            guidance_scale: guidance scale for classifier-free guidance (if implemented)
        
        Returns:
            output: (batch_size, output_size) - normalized features + latent
        """
        batch_size = x_query.shape[0]
        device = x_query.device
        
        # Start from random noise
        output = torch.randn(batch_size, self.output_size, device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        
        # DDIM sampling
        for i, t in enumerate(timesteps):
            # Predict noise
            t_batch = t.repeat(batch_size)
            predicted_noise = self.predict_noise(x_query, t_batch)
            
            # Compute alpha and sigma for this timestep
            alpha_t = 1.0 - (t.float() / self.num_timesteps)
            alpha_t_prev = 1.0 - (timesteps[i-1].float() / self.num_timesteps) if i > 0 else 1.0
            
            # Predict x0
            predicted_x0 = output - predicted_noise * (1.0 - alpha_t).sqrt()
            
            # Compute direction pointing to x_t
            pred_dir = (1.0 - alpha_t_prev).sqrt() * predicted_noise
            
            # Update output
            output = alpha_t_prev.sqrt() * predicted_x0 + pred_dir
        
        return output


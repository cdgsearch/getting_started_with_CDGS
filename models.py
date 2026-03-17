import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLPBlock(nn.Module):
    """MLP block with time conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First layer with time conditioning
        x = self.norm1(x)
        x = F.silu(self.linear1(x) + self.time_proj(time_emb))
        x = self.dropout(x)
        
        # Second layer
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class Simple2DUNet(nn.Module):
    """
    Simple UNet-like architecture for 2D diffusion model.
    Takes 2D points and timestep, outputs noise prediction.
    """
    
    def __init__(
        self, 
        input_dim: int = 2,
        hidden_dims: tuple[int, ...] = (128, 256, 512),
        time_embed_dim: int = 128,
        num_blocks_per_level: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            blocks = nn.ModuleList([
                MLPBlock(hidden_dim, hidden_dim * 2, time_embed_dim, dropout)
                for _ in range(num_blocks_per_level)
            ])
            self.encoder_blocks.append(blocks)
            
            if i < len(hidden_dims) - 1:
                self.encoder_downs.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
        
        # Middle block
        self.middle_block = MLPBlock(hidden_dims[-1], hidden_dims[-1] * 2, time_embed_dim, dropout)
        
        # Decoder (upsampling)
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        reversed_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims[:-1]):
            next_dim = reversed_dims[i + 1]
            self.decoder_ups.append(nn.Linear(hidden_dim, next_dim))
            
            # After concatenation with skip connection, we have next_dim * 2
            # We need to project it back to next_dim for the residual connection
            concat_proj = nn.Linear(next_dim * 2, next_dim)
            blocks = nn.ModuleList([
                concat_proj,  # Project concatenated features back to next_dim
                *[MLPBlock(next_dim, next_dim * 2, time_embed_dim, dropout) 
                  for _ in range(num_blocks_per_level)]
            ])
            self.decoder_blocks.append(blocks)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[0], input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 2)
            timestep: Timestep tensor of shape (batch_size,)
            
        Returns:
            Noise prediction of shape (batch_size, 2)
        """
        # Time embedding
        time_emb = self.time_embedding(timestep)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder with skip connections
        skip_connections = []
        for blocks, down in zip(self.encoder_blocks[:-1], self.encoder_downs):
            for block in blocks:
                x = block(x, time_emb)
            skip_connections.append(x)
            x = F.silu(down(x))
        
        # Last encoder block (no downsampling)
        for block in self.encoder_blocks[-1]:
            x = block(x, time_emb)
        
        # Middle block
        x = self.middle_block(x, time_emb)

        # print("Middle block output shape:", x.shape)
        
        # Decoder with skip connections
        for up, blocks, skip in zip(self.decoder_ups, self.decoder_blocks, reversed(skip_connections)):
            x = F.silu(up(x))
            # print("Decoder upsample output shape:", x.shape)
            # print("Skip connection shape:", skip.shape)
            x = torch.cat([x, skip], dim=-1)  # Skip connection
            # print("After concatenation shape:", x.shape)
            
            # First block is the projection layer
            concat_proj = blocks[0]
            x = concat_proj(x)
            # print("After projection shape:", x.shape)
            
            # Apply remaining MLP blocks
            for block in blocks[1:]:
                x = block(x, time_emb)
            # print("Decoder block output shape:", x.shape)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class SimpleDiffusionModel(nn.Module):
    """
    Complete diffusion model with DDPM scheduler integration.
    """
    
    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        input_dim: int = 2,
        **unet_kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        
        if unet is None:
            self.unet = Simple2DUNet(input_dim=input_dim, **unet_kwargs)
        else:
            self.unet = unet
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.unet(x, timestep)
    
    def training_step(self, batch: torch.Tensor, noise_scheduler) -> torch.Tensor:
        """
        Training step for diffusion model.
        
        Args:
            batch: Clean data of shape (batch_size, input_dim)
            noise_scheduler: DDPM scheduler from diffusers
            
        Returns:
            Loss tensor
        """
        batch_size = batch.shape[0]
        device = batch.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(batch)
        
        # Add noise to clean images according to timestep
        noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        noise_pred = self(noisy_batch, timesteps)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        noise_scheduler,
        device: torch.device,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling.
        
        Args:
            batch_size: Number of samples to generate
            noise_scheduler: DDPM scheduler from diffusers
            device: Device to run on
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated samples of shape (batch_size, input_dim)
        """
        # Start with random noise
        sample = torch.randn((batch_size, self.input_dim), device=device)
        
        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self(sample, t.expand(batch_size).to(device))
            
            # Denoise
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
        
        return sample
    
class FlowMatchingModel(nn.Module):
    """
    Complete flow matching model.
    """
    
    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        input_dim: int = 2,
        **unet_kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        
        if unet is None:
            self.unet = Simple2DUNet(input_dim=input_dim, **unet_kwargs)
        else:
            self.unet = unet
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass returns the predicted velocity vector."""
        return self.unet(x, timestep)
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Training step for Conditional Flow Matching.

        Uses the convention:
            x_sigma = (1 - sigma) * x_data + sigma * x_noise
            v = x_noise - x_data   (velocity points toward noise)

        The model takes sigma (noise level in [0, 1]) as the time input.
        At sigma=0: clean data.  At sigma=1: pure noise.

        Args:
            batch: Clean data of shape (batch_size, input_dim)

        Returns:
            Loss tensor
        """
        x_data = batch
        x_noise = torch.randn_like(x_data)

        # sigma ~ U[0, 1] represents noise level
        sigma = torch.rand(x_data.shape[0], device=x_data.device).unsqueeze(-1)

        # Interpolate: x_sigma = (1 - sigma) * data + sigma * noise
        x_sigma = (1 - sigma) * x_data + sigma * x_noise

        # Target velocity: v = noise - data (points toward noise, matching SD3/FLUX)
        u_target = x_noise - x_data

        # Predict the velocity v(x_sigma, sigma)
        v_pred = self(x_sigma, sigma.squeeze(-1))

        loss = F.mse_loss(v_pred, u_target)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        num_inference_steps: int = 100
    ) -> torch.Tensor:
        """
        Generate samples using Euler ODE integration.

        Uses sigma schedule [1.0 -> 0.0] (noise -> data).  
        v = noise - data, so:
            x_{sigma+d_sigma} = x_sigma + v * d_sigma
        with d_sigma < 0 gives movement toward data.

        Args:
            batch_size: Number of samples to generate
            device: Device to run on
            num_inference_steps: Number of integration steps

        Returns:
            Generated samples of shape (batch_size, input_dim)
        """
        sample = torch.randn((batch_size, self.input_dim), device=device)

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            dt = sigmas[i + 1] - sigma  # negative (denoising)
            velocity = self(sample, sigma.expand(batch_size))
            sample = sample + velocity * dt

        return sample
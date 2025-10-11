import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import DDPMScheduler
from utils import load_models
from diffusers.utils.torch_utils import randn_tensor
from models import SimpleDiffusionModel, FlowMatchingModel


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Create sliding window views
def create_views(num_models):
    """
    Create sliding 2D views over the latent space
    For M models, views are (0,2), (1,3), ..., (M-1, M+1)
    Total latent dimension = M+1
    """
    views = [(i, i + 2) for i in range(num_models)]
    print(f"Created {len(views)} views: {views}")
    print(f"Total latent dimension: {views[-1][1]}")
    return views

def undo_step(latents, timestep, scheduler, generator=None):
    """
    Reverse a single diffusion denoising step to return to a noisier state.
    
    This function is used in the iterative resampling procedure to "undo" a 
    denoising step, allowing the sampler to re-denoise from the same timestep
    multiple times for better compositional alignment.
    
    Args:
        latents: current latent state (B, dim)
        timestep: the current timestep in the scheduler
        scheduler: DDPMScheduler instance
        generator: optional random generator for reproducibility
        
    Returns:
        latents: noisier latent state (one step back in the forward process)
    """
    n = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    for i in range(n):
        beta = scheduler.betas[timestep + i]
        if latents.device.type == "mps":
            # randn does not work reproducibly on mps
            noise = randn_tensor(latents.shape, dtype=latents.dtype, generator=generator)
            noise = noise.to(latents.device)
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

        # Apply forward diffusion step: x_t = sqrt(1-beta) * x_{t-1} + sqrt(beta) * noise
        latents = (1 - beta) ** 0.5 * latents + beta**0.5 * noise

    return latents


def compute_inversion_scores(batched_x0s, views, models, scheduler, device):
    """
    Compute smoothness scores for predicted clean samples using DDIM inversion.
    
    This function performs DDIM inversion on each local plan segment to measure
    how smoothly each sample inverts back to noise. Smoother inversions (lower
    scores) indicate more plausible/consistent local plan segments.
    
    The score is computed as the L2 norm of the rate of change of predicted noise
    along the inversion trajectory.
    
    Args:
        batched_x0s: clean predictions for each view, shape [num_models, B, local_dim]
        views: list of (start, end) tuples for each local plan
        models: list of model instances
        scheduler: DDPMScheduler instance
        device: torch device
        
    Returns:
        final_scores: smoothness scores for each sample, shape (B,)
    """
    num_models = len(views)
    B = batched_x0s.shape[1]
    
    # Get timesteps for inversion (reverse the scheduler order)
    all_timesteps = scheduler.timesteps.flip(dims=(0,))
    num_timesteps = len(all_timesteps)
    
    # Initialize inversion from clean predictions
    inversion_latents = batched_x0s.clone()
    all_noise_prediction = []
    
    # Perform DDIM inversion for a fraction of timesteps (5% by default)
    # We only need a short inversion to assess smoothness
    inversion_steps = num_timesteps // 20
    for idx in tqdm(range(inversion_steps), leave=False, desc="Computing inversion scores"):
        t = all_timesteps[idx]
        t_next = all_timesteps[idx + 1]
        
        # Get alpha values for DDIM inversion formula
        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_next = scheduler.alphas_cumprod[t_next]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

        with torch.no_grad():
            # Get noise predictions from each model for its corresponding view
            noise_pred_combined = torch.zeros_like(inversion_latents)
            for id, (start, end) in enumerate(views):
                latent_view = inversion_latents[id]
                noise_pred = models[id](latent_view, t.repeat(B).to(device))
                noise_pred_combined[id] = noise_pred

            # DDIM inversion: predict x0, then predict next noisy state
            x0_pred = (inversion_latents - sqrt_one_minus_alpha_t * noise_pred_combined) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            noise_pred_combined = (inversion_latents - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t
            inversion_latents = sqrt_alpha_t_next * x0_pred + sqrt_one_minus_alpha_t_next * noise_pred_combined
            
            all_noise_prediction.append(noise_pred_combined)

    # Stack predictions and compute derivatives
    all_intermediate_noise_preds = torch.stack(all_noise_prediction, dim=2)  # [num_models, B, timesteps, local_dim]
    derivative = torch.diff(all_intermediate_noise_preds, dim=2)
    
    # Compute scores: L2 norm of derivative for each sample, averaged across models
    all_scores = torch.norm(derivative.reshape(num_models * B, -1), dim=1).reshape(num_models, B)
    final_scores = all_scores.mean(dim=0)  # (B,)
    
    return final_scores


def rearrange_batch_by_scores(latents, scores, top_K):
    """
    Rearrange batch by selecting top-K samples with lowest scores and replicating.
    
    Args:
        latents: current latent batch (B, dim)
        scores: smoothness scores for each sample (B,)
        top_K: fraction of samples to retain (0 < top_K <= 1)
        
    Returns:
        arranged_batch: rearranged latent batch with same shape (B, dim)
    """
    B = latents.shape[0]
    num_selected_samples = max(int(top_K * B), 1)
    
    # Select indices with lowest scores (smoothest inversions)
    topk_indices = torch.topk(scores, k=num_selected_samples, largest=False)[1]
    
    # Create new batch from selected samples
    arranged_batch = latents[topk_indices].clone()
    
    # Replicate selected samples to fill batch
    while arranged_batch.shape[0] < B:
        arranged_batch = torch.cat([arranged_batch, arranged_batch], dim=0)
    
    # Trim to exact batch size
    arranged_batch = arranged_batch[:B]
    
    return arranged_batch


# =============================================================================
# MAIN CDGS SAMPLER CLASS
# =============================================================================

class CDGS(nn.Module):
    """
    Compositional Diffusion Guidance Sampler (CDGS) for long-horizon planning.
    
    This sampler implements compositional generation by maintaining multiple local
    models that each predict over sliding windows of a long latent sequence. The
    predictions are averaged in overlapping regions to create a coherent long-range
    sample.
    
    Key Features:
    - Compositional prediction: Multiple models cover overlapping windows
    - Iterative resampling: Re-denoise from same timestep for better alignment
    - Inversion pruning: Select samples with smoother inversion paths
    
    Parameters:
        model_paths: Dictionary of model checkpoint paths for loading models
        device: torch device where models and sampling will run
        model_type: 'diffusion' or 'flow' (flow support is scaffolded for future)
        num_bridges: Number of bridge models defining the compositional task
        
        num_resampling_steps: Number of times to denoise from each timestep
            - 1 (default): No iterative resampling, standard compositional sampling
            - >1: Enable iterative resampling for better compositional alignment
            
        enable_pruning: Whether to apply inversion-based pruning
            - False (default): No pruning, standard resampling
            - True: Apply pruning to select smoother samples during generation
            
        pruning_start: Fraction of sampling steps before pruning starts (0-1)
        pruning_end: Fraction of sampling steps after pruning ends (0-1)
        pruning_top_K: Fraction of batch to retain during pruning (0-1)
    
    Usage Examples:
        # Basic CDGS (no resampling, no pruning)
        sampler = CDGS(model_paths, device)
        
        # CDGS with resampling only
        sampler = CDGS(model_paths, device, num_resampling_steps=5)
        
        # Full CDGS with resampling and pruning
        sampler = CDGS(model_paths, device, num_resampling_steps=5, enable_pruning=True)
    """

    def __init__(
        self, 
        model_paths, 
        device, 
        model_type='diffusion', 
        num_bridges=1,
        num_resampling_steps=1,
        enable_pruning=False,
        pruning_start=0.1, 
        pruning_end=0.9, 
        pruning_top_K=0.2
    ):
        """Initialize the CDGS sampler with specified configuration."""
        super().__init__()
        self.device = device
        self.model_type = model_type
        
        # Resampling and pruning parameters
        self.num_resampling_steps = num_resampling_steps
        self.enable_pruning = enable_pruning
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_top_K = pruning_top_K

        # Load models and create sliding window views
        self.models = load_models(device, model_paths, model_type, num_bridges)
        if self.models:
            self.views = create_views(len(self.models))
            self.latent_dim = self.views[-1][1]
        else:
            raise ValueError("Models could not be loaded. Please check model paths.")

        # Initialize DDPM scheduler for timesteps and noise schedules
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
        )

        # Print configuration summary
        print("✅ CDGS initialized:")
        print(f"  Model type: {self.model_type}")
        print(f"  Number of models: {len(self.models)}")
        print(f"  Views: {self.views}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Resampling steps: {self.num_resampling_steps}" + 
              (" (disabled)" if self.num_resampling_steps == 1 else ""))
        print(f"  Pruning: {'enabled' if self.enable_pruning else 'disabled'}")
        if self.enable_pruning:
            print(f"    - Pruning window: {self.pruning_start} to {self.pruning_end}")
            print(f"    - Top-K fraction: {self.pruning_top_K}")

    def get_compositional_prediction(self, latent, t):
        """
        Compute compositional prediction by averaging overlapping view outputs.
        
        For each sliding window view over the long latent sequence, the corresponding
        model predicts a residual/noise term. We accumulate predictions and normalize
        by the cover count to obtain a combined prediction over the full sequence.
        
        Args:
            latent: current latent state (B, latent_dim)
            t: current timestep (scalar or tensor)
            
        Returns:
            combined_pred: compositional prediction (B, latent_dim)
        """
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        B = latent.shape[0]

        # Ensure time is a tensor with shape (B,)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=latent.dtype, device=self.device)
        t_vec = t.reshape(1).repeat(B).to(self.device)

        # Iterate through views and aggregate predictions
        for id, (start, end) in enumerate(self.views):
            latent_view = latent[:, start:end]
            pred = self.models[id](latent_view, t_vec)

            value[:, start:end] += pred
            count[:, start:end] += 1

        # Average predictions in overlapping regions
        combined_pred = torch.where(count > 0, value / count, value)
        return combined_pred

    def inversion_pruning(self, pred_x0, latents, return_scores=False):
        """
        Prune batch by selecting samples with smoothest inversion paths.
        
        This method uses DDIM inversion to assess the smoothness of each sample's
        trajectory when inverted back toward noise. Samples with smoother inversions
        (lower derivative norms) are interpreted as more plausible and are retained.
        
        Args:
            pred_x0: predicted clean samples (B, latent_dim)
            latents: current noisy latents (B, latent_dim)
            return_scores: if True, return scores along with rearranged batch
            
        Returns:
            arranged_batch: rearranged latent batch (B, latent_dim)
            final_scores (optional): inversion scores for each sample (B,)
        """
        B = pred_x0.shape[0]
        
        # Split predicted clean samples into local views
        batched_x0s = []
        for start, end in self.views:
            batched_x0s.append(pred_x0[:, start:end])
        batched_x0s = torch.stack(batched_x0s, dim=0)  # [num_models, B, local_dim]

        # Compute smoothness scores via DDIM inversion
        final_scores = compute_inversion_scores(
            batched_x0s, self.views, self.models, self.scheduler, self.device
        )
        
        # Rearrange batch by selecting smoothest samples
        arranged_batch = rearrange_batch_by_scores(latents, final_scores, self.pruning_top_K)

        if return_scores:
            return arranged_batch, final_scores
        return arranged_batch

    @torch.no_grad()
    def sample(self, batch_size=100, num_inference_steps=100):
        """
        Generate samples using compositional diffusion with optional resampling and pruning.
        
        The sampling process consists of:
        1. Initialize from random noise
        2. For each timestep:
           a. Denoise using compositional predictions
           b. (Optional) Apply pruning to select better samples
           c. (Optional) Undo the step and re-denoise for better alignment
        3. Return final samples
        
        Args:
            batch_size: number of trajectories to sample
            num_inference_steps: number of denoising steps
            
        Returns:
            latent: sampled sequences (batch_size, latent_dim)
        """
        # Initialize from random noise
        sequence_dim = self.views[-1][1]
        latent = torch.randn((batch_size, sequence_dim)).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        num_timesteps = len(self.scheduler.timesteps)

        with torch.autocast(device_type=self.device):
            # Main denoising loop
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                
                # Iterative resampling loop (runs once if num_resampling_steps=1)
                for u in range(self.num_resampling_steps):
                    
                    # ============================================================
                    # DENOISING STEP
                    # ============================================================
                    if self.model_type == SimpleDiffusionModel:
                        # Get compositional noise prediction
                        eps_pred = self.get_compositional_prediction(latent, t)
                        
                        # Take denoising step
                        output = self.scheduler.step(eps_pred, t, latent)
                        pred_x0 = output.pred_original_sample
                        latent = output.prev_sample
                        
                    else:  # flow model
                        # Flow model sampling is scaffolded for future implementation
                        # Flow models require different sampling equations (ODE/SDE integration)
                        # rather than the epsilon-prediction + scheduler.step() approach
                        raise NotImplementedError(
                            "Flow model sampling not implemented yet. "
                            "Would require ODE/SDE solver instead of DDPM scheduler."
                        )

                    # ============================================================
                    # PRUNING STEP (if enabled and in pruning window)
                    # ============================================================
                    # Apply pruning on second-to-last resampling iteration
                    if self.enable_pruning and u == self.num_resampling_steps - 2:
                        # Check if we're in the pruning window
                        if self.pruning_start * num_timesteps < i < self.pruning_end * num_timesteps:
                            latent = self.inversion_pruning(pred_x0, latent)
                    
                    # ============================================================
                    # UNDO STEP (if doing iterative resampling)
                    # ============================================================
                    # Undo the denoising step to return to x_t for next resampling iteration
                    # Skip on the last resampling iteration and at boundary timesteps
                    if self.num_resampling_steps > 1 and u < self.num_resampling_steps - 1:
                        if 0 < i < len(self.scheduler.timesteps) - 1:
                            latent = undo_step(latent, t, self.scheduler)
        
        return latent

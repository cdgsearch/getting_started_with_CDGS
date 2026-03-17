"""
CDGS Samplers — Compositional Diffusion / Flow Matching Guidance Search
=========================================================================
"""

import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Type, Union
from tqdm import tqdm
from diffusers import DDPMScheduler
from utils import load_models
from diffusers.utils.torch_utils import randn_tensor
from models import SimpleDiffusionModel, FlowMatchingModel


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@beartype
def create_views(num_models: int) -> List[Tuple[int, int]]:
    """
    Create sliding 2D views over the latent space.

    For M models, views are (0,2), (1,3), ..., (M-1, M+1).
    Total latent dimension = M+1.

    Args:
        num_models: Number of local models to create views for.

    Returns:
        List of (start, end) index tuples for each view.
    """
    views = [(i, i + 2) for i in range(num_models)]
    print(f"Created {len(views)} views: {views}")
    print(f"Total latent dimension: {views[-1][1]}")
    return views


# =============================================================================
# DIFFUSION HELPER FUNCTIONS
# =============================================================================

@beartype
def undo_step(
    latents: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    scheduler: DDPMScheduler,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Reverse a single diffusion denoising step to return to a noisier state.
    Used in the iterative resampling procedure.

    Args:
        latents: Current latent state (B, dim).
        timestep: Current timestep in the scheduler.
        scheduler: DDPMScheduler instance.
        generator: Optional random generator for reproducibility.

    Returns:
        Noisier latent state (B, dim).
    """
    n = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    for i in range(n):
        beta = scheduler.betas[timestep + i]
        if latents.device.type == "mps":
            noise = randn_tensor(latents.shape, dtype=latents.dtype, generator=generator)
            noise = noise.to(latents.device)
        else:
            noise = randn_tensor(
                latents.shape, generator=generator,
                device=latents.device, dtype=latents.dtype,
            )

        latents = (1 - beta) ** 0.5 * latents + beta**0.5 * noise

    return latents


@beartype
def compute_inversion_scores(
    batched_x0s: torch.Tensor,
    views: List[Tuple[int, int]],
    models: List[nn.Module],
    scheduler: DDPMScheduler,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    Compute smoothness scores for predicted clean samples using DDIM inversion.

    Performs DDIM inversion on each local plan segment and measures the L2 norm
    of the rate of change of noise predictions along the trajectory.

    Args:
        batched_x0s: Clean predictions per view, shape [num_models, B, local_dim].
        views: List of (start, end) tuples for each local plan.
        models: List of model instances, one per view.
        scheduler: DDPMScheduler instance with set_timesteps already called.
        device: Torch device.

    Returns:
        Smoothness scores for each sample, shape (B,). Lower = better.
    """
    num_models = len(views)
    B = batched_x0s.shape[1]

    all_timesteps = scheduler.timesteps.flip(dims=(0,))
    num_timesteps = len(all_timesteps)

    inversion_latents = batched_x0s.clone()
    all_noise_prediction = []

    inversion_steps = num_timesteps // 20
    for idx in tqdm(range(inversion_steps), leave=False, desc="Computing inversion scores"):
        t = all_timesteps[idx]
        t_next = all_timesteps[idx + 1]

        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_next = scheduler.alphas_cumprod[t_next]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

        with torch.no_grad():
            noise_pred_combined = torch.zeros_like(inversion_latents)
            for id, (start, end) in enumerate(views):
                latent_view = inversion_latents[id]
                noise_pred = models[id](latent_view, t.repeat(B).to(device))
                noise_pred_combined[id] = noise_pred

            x0_pred = (inversion_latents - sqrt_one_minus_alpha_t * noise_pred_combined) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            noise_pred_combined = (inversion_latents - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t
            inversion_latents = sqrt_alpha_t_next * x0_pred + sqrt_one_minus_alpha_t_next * noise_pred_combined

            all_noise_prediction.append(noise_pred_combined)

    all_intermediate_noise_preds = torch.stack(all_noise_prediction, dim=2)
    derivative = torch.diff(all_intermediate_noise_preds, dim=2)

    all_scores = torch.norm(derivative.reshape(num_models * B, -1), dim=1).reshape(num_models, B)
    final_scores = all_scores.mean(dim=0)

    return final_scores


# =============================================================================
# FLOW MATCHING HELPER FUNCTIONS
#
# Convention:
#   x_sigma = (1 - sigma) * x_data + sigma * x_noise
#   v = x_noise - x_data   (points toward noise)
#   sigma in [1, 0]:  1 = pure noise,  0 = clean data
#   Euler step:  x_{sigma+ds} = x_sigma + v * ds,  ds < 0  ->  toward data
# =============================================================================

@beartype
def flow_matching_euler_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    dt: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Euler ODE step: x_{sigma+dt} = x_sigma + v * dt.

    With v = noise - data and dt < 0 (denoising), this moves toward data.

    Args:
        latent: Current state x_sigma (B, dim).
        velocity: Predicted velocity v_sigma (B, dim).
        dt: Time delta. Negative for denoising (sigma_next - sigma < 0).

    Returns:
        Next state x_{sigma+dt} (B, dim).
    """
    return latent + velocity * dt


@beartype
def flow_matching_predict_x0(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Estimate clean data x_0 from current state and velocity.

    x_sigma = (1 - sigma) * x_0 + sigma * noise
    v = noise - x_0
    => x_sigma = x_0 + sigma * v
    => x_0 = x_sigma - v * sigma

    Args:
        latent: Current state x_sigma (B, dim).
        velocity: Predicted velocity v_sigma (B, dim).
        sigma: Current noise level (scalar or tensor in [0, 1]).

    Returns:
        Estimated clean sample x_0 (B, dim).
    """
    return latent - velocity * sigma


@beartype
def flow_matching_undo_step(
    latent: torch.Tensor,
    pred_x0: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    sigma_next: Union[float, torch.Tensor],
    eta: float = 0.5,
) -> torch.Tensor:
    """
    Undo step for flow matching resampling.

    Decomposes x_{sigma_next} into signal (pred_x0) and noise, then
    recombines at the noisier sigma level with partial noise refresh.

    x_sigma = (1 - sigma) * x_0 + sigma * epsilon

    Args:
        latent: Current denoised state x_{sigma_next} (B, dim).
        pred_x0: Predicted clean sample x_0 (B, dim).
        sigma: Target noise level to return to (noisier, larger).
        sigma_next: Current noise level (less noisy, smaller).
        eta: Noise refresh fraction in [0, 1].
             0 = deterministic (keep existing noise direction).
             1 = fully stochastic (resample all noise).

    Returns:
        Noisier latent state at sigma level (B, dim).
    """
    if sigma_next > 1e-6:
        eps_current = (latent - (1 - sigma_next) * pred_x0) / sigma_next

        eps_fresh = torch.randn_like(latent)
        eps_mixed = (1 - eta) * eps_current + eta * eps_fresh

        renorm = 1.0 / ((1 - eta) ** 2 + eta ** 2) ** 0.5
        eps_mixed = eps_mixed * renorm

        return (1 - sigma) * pred_x0 + sigma * eps_mixed
    else:
        eps_fresh = torch.randn_like(latent)
        return (1 - sigma) * pred_x0 + sigma * eps_fresh


@beartype
def compute_flow_inversion_scores(
    batched_x0s: torch.Tensor,
    views: List[Tuple[int, int]],
    models: List[nn.Module],
    device: Union[str, torch.device],
    total_steps: int = 100,
) -> torch.Tensor:
    """
    Compute smoothness scores via forward ODE integration for flow matching.

    Starting from predicted clean samples (x_0), integrate forward (sigma 0->0.5)
    and measure velocity smoothness. Lower scores = better.

    Args:
        batched_x0s: Predicted clean samples per view [num_models, B, local_dim].
        views: List of (start, end) tuples for each local plan.
        models: List of flow matching model instances, one per view.
        device: Torch device.
        total_steps: Total number of sampling steps (determines num_inv_steps).

    Returns:
        Smoothness scores for each sample, shape (B,). Lower = better.
    """
    num_models = len(views)
    B = batched_x0s.shape[1]

    num_inv_steps = max(3, total_steps // 20)

    inv_sigmas = torch.linspace(0.0, 0.5, num_inv_steps + 1, device=device)

    inversion_latents = batched_x0s.clone()
    all_velocity_preds = []

    with torch.no_grad():
        for idx in range(num_inv_steps):
            sigma = inv_sigmas[idx]
            sigma_next = inv_sigmas[idx + 1]
            dt = sigma_next - sigma

            velocity_combined = torch.zeros_like(inversion_latents)
            for id, (start, end) in enumerate(views):
                latent_view = inversion_latents[id]
                velocity = models[id](latent_view, sigma.repeat(B).to(device))
                velocity_combined[id] = velocity

            inversion_latents = inversion_latents + velocity_combined * dt
            all_velocity_preds.append(velocity_combined)

    all_preds = torch.stack(all_velocity_preds, dim=2)
    derivative = torch.diff(all_preds, dim=2)

    all_scores = torch.norm(derivative.reshape(num_models * B, -1), dim=1).reshape(num_models, B)
    final_scores = all_scores.mean(dim=0)

    return final_scores


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

@beartype
def compute_adaptive_top_k(
    scores: torch.Tensor,
    base_top_k: float,
    cv_threshold: float = 0.15,
) -> int:
    """
    CV-based adaptive pruning.

    Adjusts how many samples to keep based on the coefficient of variation
    (CV = std / mean) of inversion scores:
    - High CV (spread scores)    -> prune aggressively toward base_top_k
    - Low  CV (clustered scores) -> keep most samples (pruning near-random)

    Args:
        scores: Inversion scores for each sample, shape (B,). Lower = better.
        base_top_k: Target keep fraction at full pruning strength in (0, 1].
        cv_threshold: CV value at which pruning reaches ~50% strength.

    Returns:
        Number of samples to retain.
    """
    B = scores.shape[0]
    score_mean = scores.mean()
    score_std = scores.std()
    cv = (score_std / score_mean).item() if score_mean.abs().item() > 1e-8 else 0.0
    ratio = min(1.0, (cv / cv_threshold) ** 2) if cv_threshold > 0 else 1.0
    keep_frac = 1.0 - ratio * (1.0 - base_top_k)
    num_keep = max(1, int(keep_frac * B))
    return num_keep


@beartype
def compute_U(
    step_idx: int,
    total_steps: int,
    max_U: int,
    min_U: int = 1,
) -> int:
    """
    Compute number of resampling iterations U for the current step.
    Linearly ramps from min_U (early steps) to max_U (later steps).

    Args:
        step_idx: Current denoising step index.
        total_steps: Total number of denoising steps.
        max_U: Maximum resampling iterations.
        min_U: Minimum resampling iterations.

    Returns:
        Number of resampling iterations for this step.
    """
    if max_U <= min_U:
        return min_U
    frac = step_idx / max(total_steps - 1, 1)
    return int(min(max(frac * max_U, min_U), max_U))


@beartype
def rearrange_batch_by_scores(
    latents: torch.Tensor,
    scores: torch.Tensor,
    num_keep: int,
) -> torch.Tensor:
    """
    Select top-k samples with lowest scores (best quality) and replicate
    to fill the original batch size.

    Args:
        latents: Current latent batch (B, dim).
        scores: Per-sample scores (B,). Lower = better.
        num_keep: Number of top samples to retain.

    Returns:
        Rearranged latent batch with original batch size (B, dim).
    """
    B = latents.shape[0]
    topk_indices = torch.topk(scores, k=num_keep, largest=False)[1]
    arranged = latents[topk_indices].clone()
    while arranged.shape[0] < B:
        arranged = torch.cat([arranged, arranged], dim=0)
    return arranged[:B]


# =============================================================================
# MAIN CDGS SAMPLER CLASS
# =============================================================================

class CDGS(nn.Module):
    """
    Compositional Diffusion/Flow Matching Guidance Sampler (CDGS).

    Implements compositional generation by maintaining multiple local models
    that each predict over sliding windows of a long latent sequence. Predictions
    are averaged in overlapping regions. Supports both diffusion (DDPM) and
    flow matching models.
    """

    @beartype
    def __init__(
        self,
        model_paths: Dict[str, str],
        device: str,
        model_type: Type[nn.Module] = SimpleDiffusionModel,
        num_bridges: int = 1,
        num_resampling_steps: int = 1,
        min_resampling_steps: int = 1,
        enable_pruning: bool = False,
        pruning_start: float = 0.1,
        pruning_end: float = 0.9,
        pruning_top_K: float = 0.2,
        adaptive_pruning: bool = True,
        cv_threshold: float = 0.15,
        undo_eta: float = 0.5,
    ):
        """
        Args:
            model_paths: Dictionary with keys 'start', 'bridge', 'end' mapping
                to checkpoint file paths.
            device: Torch device string (e.g. 'cuda', 'cpu', 'mps').
            model_type: Model class — SimpleDiffusionModel or FlowMatchingModel.
            num_bridges: Number of bridge models between start and end.
            num_resampling_steps: Max resampling iterations per timestep (U).
            min_resampling_steps: Min resampling iterations (ramps from min to max).
            enable_pruning: Whether to apply inversion-based pruning.
            pruning_start: Fraction of steps before pruning starts (0-1).
            pruning_end: Fraction of steps after which pruning stops (0-1).
            pruning_top_K: Base keep fraction for pruning (0-1).
            adaptive_pruning: Enable CV-based adaptive pruning.
            cv_threshold: CV threshold for adaptive pruning.
            undo_eta: Noise refresh fraction for flow matching undo step (0-1).
        """
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.is_flow = (model_type == FlowMatchingModel)

        self.num_resampling_steps = num_resampling_steps
        self.min_resampling_steps = min_resampling_steps
        self.enable_pruning = enable_pruning
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_top_K = pruning_top_K
        self.adaptive_pruning = adaptive_pruning
        self.cv_threshold = cv_threshold
        self.undo_eta = undo_eta

        self.models = load_models(device, model_paths, model_type, num_bridges)
        if self.models:
            self.views = create_views(len(self.models))
            self.latent_dim = self.views[-1][1]
        else:
            raise ValueError("Models could not be loaded. Please check model paths.")

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
        )

        mode_str = "flow matching" if self.is_flow else "diffusion"
        print(f"CDGS initialized ({mode_str}):")
        print(f"  Number of models: {len(self.models)}")
        print(f"  Views: {self.views}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Resampling steps: {self.num_resampling_steps}" +
              (" (disabled)" if self.num_resampling_steps <= 1 else
               f" (min={self.min_resampling_steps})"))
        print(f"  Pruning: {'enabled' if self.enable_pruning else 'disabled'}")
        if self.enable_pruning:
            print(f"    - Pruning window: {self.pruning_start} to {self.pruning_end}")
            print(f"    - Top-K fraction: {self.pruning_top_K}")
            print(f"    - Adaptive: {self.adaptive_pruning} (CV threshold={self.cv_threshold})")
        if self.is_flow:
            print(f"  Undo eta: {self.undo_eta}")

    @beartype
    def get_compositional_prediction(
        self,
        latent: torch.Tensor,
        t: Union[float, int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute compositional prediction by averaging overlapping view outputs.
        For diffusion: noise prediction. For flow matching: velocity prediction.

        Args:
            latent: Current latent state (B, latent_dim).
            t: Current timestep (scalar, int, or tensor).

        Returns:
            Compositional noise/velocity prediction (B, latent_dim).
        """
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        B = latent.shape[0]

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=latent.dtype, device=self.device)
        t_vec = t.reshape(1).repeat(B).to(self.device)

        for id, (start, end) in enumerate(self.views):
            latent_view = latent[:, start:end]
            pred = self.models[id](latent_view, t_vec)

            value[:, start:end] += pred
            count[:, start:end] += 1

        combined_pred = torch.where(count > 0, value / count, value)
        return combined_pred

    @beartype
    def inversion_pruning(
        self,
        pred_x0: torch.Tensor,
        latents: torch.Tensor,
        num_inference_steps: int = 100,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prune batch by selecting samples with smoothest inversion paths.
        Uses adaptive pruning (CV-based) to avoid mode collapse.

        Args:
            pred_x0: Predicted clean samples (B, latent_dim).
            latents: Current noisy latents (B, latent_dim).
            num_inference_steps: Total inference steps (used for flow inversion).
            return_scores: If True, also return the raw inversion scores.

        Returns:
            Rearranged latent batch (B, latent_dim), and optionally scores (B,).
        """
        batched_x0s = []
        for start, end in self.views:
            batched_x0s.append(pred_x0[:, start:end])
        batched_x0s = torch.stack(batched_x0s, dim=0)

        if self.is_flow:
            final_scores = compute_flow_inversion_scores(
                batched_x0s, self.views, self.models, self.device,
                total_steps=num_inference_steps,
            )
        else:
            final_scores = compute_inversion_scores(
                batched_x0s, self.views, self.models, self.scheduler, self.device
            )

        if self.adaptive_pruning:
            num_keep = compute_adaptive_top_k(
                final_scores, self.pruning_top_K, cv_threshold=self.cv_threshold
            )
        else:
            num_keep = max(1, int(self.pruning_top_K * latents.shape[0]))

        arranged_batch = rearrange_batch_by_scores(latents, final_scores, num_keep)

        if return_scores:
            return arranged_batch, final_scores
        return arranged_batch

    @beartype
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 100,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """
        Generate samples using compositional diffusion or flow matching
        with optional resampling and pruning.

        Args:
            batch_size: Number of trajectories to sample.
            num_inference_steps: Number of denoising/ODE steps.

        Returns:
            Sampled sequences (batch_size, latent_dim).
        """
        sequence_dim = self.views[-1][1]
        latent = torch.randn((batch_size, sequence_dim)).to(self.device)

        if self.is_flow:
            return self._sample_flow(latent, batch_size, num_inference_steps)
        else:
            return self._sample_diffusion(latent, batch_size, num_inference_steps)

    @beartype
    def _sample_diffusion(
        self,
        latent: torch.Tensor,
        batch_size: int,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """
        Diffusion sampling loop with DDPM scheduler.

        Args:
            latent: Initial noise (batch_size, latent_dim).
            batch_size: Number of samples.
            num_inference_steps: Number of denoising steps.

        Returns:
            Denoised samples (batch_size, latent_dim).
        """
        self.scheduler.set_timesteps(num_inference_steps)
        num_timesteps = len(self.scheduler.timesteps)

        with torch.autocast(device_type=self.device):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling (diffusion)")):
                for u in range(self.num_resampling_steps):
                    eps_pred = self.get_compositional_prediction(latent, t)
                    output = self.scheduler.step(eps_pred, t, latent)
                    pred_x0 = output.pred_original_sample
                    latent = output.prev_sample

                    if (self.enable_pruning
                            and u == self.num_resampling_steps - 2
                            and self.num_resampling_steps > 1):
                        if self.pruning_start * num_timesteps < i < self.pruning_end * num_timesteps:
                            latent = self.inversion_pruning(pred_x0, latent, num_inference_steps)

                    if self.num_resampling_steps > 1 and u < self.num_resampling_steps - 1:
                        if 0 < i < len(self.scheduler.timesteps) - 1:
                            latent = undo_step(latent, t, self.scheduler)

        return latent

    @beartype
    def _sample_flow(
        self,
        latent: torch.Tensor,
        batch_size: int,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """
        Flow matching sampling loop with Euler ODE integration.

        Uses sigma schedule [1.0 -> 0.0] (noise -> data).

        Args:
            latent: Initial noise (batch_size, latent_dim).
            batch_size: Number of samples.
            num_inference_steps: Number of ODE steps.

        Returns:
            Generated samples (batch_size, latent_dim).
        """
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        num_steps = len(sigmas) - 1

        for i in tqdm(range(num_steps), desc="Sampling (flow matching)"):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            dt = sigma_next - sigma

            U = compute_U(i, num_steps, self.num_resampling_steps,
                          self.min_resampling_steps)

            for u in range(U):
                velocity = self.get_compositional_prediction(latent, sigma)
                latent_denoised = flow_matching_euler_step(latent, velocity, dt)
                pred_x0 = flow_matching_predict_x0(latent, velocity, sigma)

                in_pruning_window = (self.pruning_start * num_steps < i
                                     < self.pruning_end * num_steps)

                if U > 1 and u < U - 1 and i < num_steps - 1:
                    if (self.enable_pruning and u == U - 2
                            and in_pruning_window and batch_size > 1):
                        latent_denoised = self.inversion_pruning(
                            pred_x0, latent_denoised, num_inference_steps
                        )
                        velocity = self.get_compositional_prediction(
                            latent_denoised, sigma_next
                        )
                        pred_x0 = flow_matching_predict_x0(
                            latent_denoised, velocity, sigma_next
                        )

                    latent = flow_matching_undo_step(
                        latent_denoised, pred_x0,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        eta=self.undo_eta,
                    )
                else:
                    latent = latent_denoised

        return latent

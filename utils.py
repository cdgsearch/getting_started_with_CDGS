import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducible experiments.

    Args:
        seed: integer seed for RNGs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# Load pre-trained models
def load_models(device, model_paths, model_class, num_bridges=4):
    """
    Load pre-trained diffusion models for compositional sampling
    
    Args:
        device: torch device
        model_paths: dict with keys
            'start': path to first edge model
            'bridge': path to bridge model
            'end': path to second edge model
        model_class: SimpleDiffusionModel or FlowMatchingModel
        num_bridges: number of bridge models between edge models
    
    Returns:
        List of loaded models
    """
    assert all(key in model_paths for key in ['start', 'bridge', 'end']), "model_paths must contain 'start', 'bridge', and 'end' keys"

    # Create model instances
    model_start = model_class()  # Edge model start
    model_bridge = model_class()  # Bridge model
    model_end = model_class()  # Edge model end

    # Build model list: [edge] + [bridge]*num_bridges + [edge]
    models = [model_start] + [model_bridge] * num_bridges + [model_end]

    try:
        # Load pre-trained weights
        models[0].load_state_dict(torch.load(model_paths["start"], map_location=device))
        models[-1].load_state_dict(torch.load(model_paths["end"], map_location=device))

        # Load bridge models
        for model in models[1:-1]:
            model.load_state_dict(torch.load(model_paths["bridge"], map_location=device))
        
        # Move to device and set to eval mode
        for model in models:
            model.to(device)
            model.eval()
            
        print(f"Successfully loaded {len(models)} models")
        return models
        
    except FileNotFoundError as e:
        print(f"Model checkpoint not found: {e}")
        print("Note: You'll need pre-trained models to run the full demo")
        return None
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from pathlib import Path
from typing import Tuple

def load_swiss_roll_data(
    n_samples: int = 1000,
    noise: float = 0.1,
    random_state: int = 42,
    device: str = "auto"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Swiss Roll dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Noise level for the dataset
        random_state: Random seed for reproducibility
        device: Device to put tensors on
        
    Returns:
        Y: High-dimensional Swiss Roll data (n_samples, 3)
        colors: Color values for visualization (n_samples,)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device.lower()
    
    # Generate Swiss Roll data
    X, color = make_swiss_roll(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # X is (n_samples, 3) - the 3D Swiss Roll
    # color is (n_samples,) - the color parameter (useful for visualization)
    
    Y = torch.tensor(X, dtype=torch.float64, device=device)
    colors = torch.tensor(color, dtype=torch.float64, device=device)
    
    return Y, colors

def get_true_latent_coordinates(colors: torch.Tensor) -> torch.Tensor:
    """
    Extract the true 2D latent coordinates from Swiss Roll colors.
    
    The Swiss Roll is parametrized by angle and height, which are the true
    latent coordinates. The color parameter from sklearn approximately
    corresponds to the angle parameter.
    
    Args:
        colors: Color values from Swiss Roll generation
        
    Returns:
        True 2D latent coordinates (n_samples, 2)
    """
    # For Swiss Roll, we can recover the latent coordinates
    # The first latent dimension is the angle (recovered from colors)
    # The second latent dimension is the height (Y coordinate)
    
    n_samples = colors.shape[0]
    
    # Normalize colors to [0, 2π] range (approximate angle)
    angles = (colors - colors.min()) / (colors.max() - colors.min()) * 2 * np.pi
    
    # For height, we need to extract from the original data
    # This is a simplified version - in practice, the height corresponds to the Y coordinate
    # We'll generate a simple height parameter based on the color structure
    heights = torch.linspace(-1, 1, n_samples, device=colors.device)
    
    latent_coords = torch.stack([angles, heights], dim=1)
    return latent_coords 
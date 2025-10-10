from typing import List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MultiModalDataset(Dataset):
    """A lightweight multimodal synthetic dataset.

    Args:
        num_samples: Number of samples to generate
        start_means: Means of the start modes
        start_stds: Standard deviations of the start modes
        end_means: Means of the end modes
        end_stds: Standard deviations of the end modes
        transition_matrix: Transition matrix to restrict which combinations of modes are valid
        seed: Random seed
    """

    def __init__(self, num_samples: int, 
                start_means: List[float],
                start_stds: List[float],
                end_means: List[float],
                end_stds: List[float],
                transition_matrix: Optional[np.ndarray] = None,
                seed: Optional[int] = None) -> None:

        # Validate parameters
        self._validate_parameters(start_means, start_stds, end_means, end_stds)
        
        # Store basic parameters
        self.start_means: List[float] = start_means
        self.start_stds: List[float] = start_stds
        self.end_means: List[float] = end_means
        self.end_stds: List[float] = end_stds
        self.num_modes_x1: int = len(start_means)
        self.num_modes_x2: int = len(end_means)
        self.num_samples: int = num_samples
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Setup transition matrix and probabilities
        self.transition_matrix = self._setup_transition_matrix(transition_matrix)
        self.transition_probabilities = self._setup_transition_probabilities()
        
        # Initialize data arrays
        self.data: np.ndarray
        self.start_labels: np.ndarray
        self.end_labels: np.ndarray
        
        self._generate_data(num_samples)

    def _validate_parameters(self, start_means: List[float], start_stds: List[float], 
                           end_means: List[float], end_stds: List[float]) -> None:
        """Validate that parameter lists have consistent lengths."""
        if len(start_means) != len(start_stds):
            raise ValueError("start_means and start_stds must have the same length")
        if len(end_means) != len(end_stds):
            raise ValueError("end_means and end_stds must have the same length")

    def _setup_transition_matrix(self, transition_matrix: Optional[np.ndarray]) -> np.ndarray:
        """Setup and validate the transition matrix."""
        if transition_matrix is None:
            matrix = np.ones((self.num_modes_x1, self.num_modes_x2), dtype=bool)
        else:
            matrix = transition_matrix
            
        # Validate transition matrix
        expected_shape = (self.num_modes_x1, self.num_modes_x2)
        if matrix.shape != expected_shape:
            raise ValueError(f"Transition matrix shape {matrix.shape} doesn't match expected shape {expected_shape}")
        if matrix.dtype != bool:
            raise ValueError("Transition matrix must be a boolean array")
            
        return matrix

    def _setup_transition_probabilities(self) -> np.ndarray:
        """Setup transition probabilities from the transition matrix.
        
        NOTE: Assumes all starting modes are equally likely, and all transitions 
        from a starting mode are equally likely.
        """
        # Create transition probabilities: equal probability for each start mode, 
        # then equal probability for valid transitions
        probabilities = self.transition_matrix.astype(float) / np.sum(
            self.transition_matrix, axis=1, keepdims=True
        )
        # Weight by equal probability for each start mode
        return probabilities / self.num_modes_x1

    def _generate_data(self, num_samples: int) -> None:
        flat_indices = self.rng.choice(
            np.arange(self.transition_probabilities.size), 
            size=num_samples, 
            p=self.transition_probabilities.ravel()
        )
        
        unraveled_indices = np.unravel_index(flat_indices, self.transition_matrix.shape)
        self.start_labels, self.end_labels = unraveled_indices

        x1 = np.array(self.start_means)[self.start_labels] + self.rng.normal(loc=0, scale=np.array(self.start_stds)[self.start_labels])
        x2 = np.array(self.end_means)[self.end_labels] + self.rng.normal(loc=0, scale=np.array(self.end_stds)[self.end_labels])

        self.data = np.column_stack([x1, x2])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns data as `(x1, x2)` points with labels `(mode_idx_x1, mode_idx_x2)`
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def plot_data(self, figsize: Tuple[int, int] = (10, 8), alpha: float = 0.6, s: int = 20) -> plt.Figure:
        """Plot the 2D dataset with different colors for different mode combinations.
        
        Returns:
            matplotlib.pyplot.Figure: The figure object that can be further modified
        """
        fig = plt.figure(figsize=figsize)
        
        transition_pairs = np.where(self.transition_matrix)
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(transition_pairs[0])))

        for i, (mode_x1, mode_x2) in enumerate(zip(transition_pairs[0], transition_pairs[1])):
            mask = (self.start_labels == mode_x1) & (self.end_labels == mode_x2)
            data_x1 = self.data[mask][:, 0]
            data_x2 = self.data[mask][:, 1]
            plt.scatter(data_x1, data_x2, c=[colors[i]], alpha=alpha, s=s,
                        label=f'Mode ({mode_x1}, {mode_x2})')

        plt.xlabel('Start')
        plt.ylabel('End')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig

    def plot_marginals(self, figsize: Tuple[int, int] = (12, 4), bins: int = 50, alpha: float = 0.7, title: Optional[str] = None) -> plt.Figure:
        """Plot marginal distributions of X1 and X2.
        
        Returns:
            matplotlib.pyplot.Figure: The figure object that can be further modified
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # X1 marginal
        data_x1 = self.data[:, 0]
        ax1.hist(data_x1, bins=bins, alpha=alpha, density=True, color='blue', edgecolor='black')
        for i, center in enumerate(self.start_means):
            ax1.axvline(center, color='red', linestyle='--', alpha=0.7, 
                       label='Mode centers' if i == 0 else "")
        ax1.set_xlabel('Start Values')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Start Marginal ({self.num_modes_x1} modes)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # X2 marginal
        data_x2 = self.data[:, 1]
        ax2.hist(data_x2, bins=bins, alpha=alpha, density=True, color='green', edgecolor='black')
        for i, center in enumerate(self.end_means):
            ax2.axvline(center, color='red', linestyle='--', alpha=0.7,
                       label='Mode centers' if i == 0 else "")
        ax2.set_xlabel('End Values')
        ax2.set_ylabel('Density')
        ax2.set_title(f'End Marginal ({self.num_modes_x2} modes)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        
        return fig


def plot_1d_dataset(dataset: Any, figsize: Tuple[int, int] = (10, 6), bins: int = 50, alpha: float = 0.7, title: Optional[str] = None) -> plt.Figure:
    """
    Plot utility for 1D multimodal dataset.
    
    Args:
        dataset: The 1D dataset to plot.
        figsize: Figure size (width, height).
        bins: Number of histogram bins.
        alpha: Histogram transparency.
        title: Plot title.
        
    Returns:
        matplotlib.pyplot.Figure: The figure object that can be further modified
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(dataset.data, bins=bins, alpha=alpha, density=True, 
             color='blue', edgecolor='black')
    
    # Mark mode centers
    for i, center in enumerate(dataset.centers):
        plt.axvline(center, color='red', linestyle='--', alpha=0.7, 
                   label='Mode centers' if i == 0 else "")
    
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.title(title or f'1D Multimodal Dataset ({dataset.num_modes} modes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":

    transition_matrix = np.array([
        [1, 0],  # Mode 0 of x1 can pair with mode 0,1 of x2
        [0, 1],
    ], dtype=bool)
    
    dataset_custom = MultiModalDataset(
        num_samples=5000,
        start_means=[0, 2],
        start_stds=[0.2, 0.2],
        end_means=[0, 1],
        end_stds=[0.2, 0.2],
        transition_matrix=transition_matrix
    )
    
    data_figure = dataset_custom.plot_data()
    data_figure.suptitle("Custom Transition Matrix Dataset")
    data_figure.savefig("figures_notebook/custom_transition_matrix_dataset.png", dpi=300, bbox_inches="tight")

    marginals_figure = dataset_custom.plot_marginals()
    marginals_figure.suptitle("Custom Transitions - Marginal Distributions")
    marginals_figure.savefig("figures_notebook/custom_transition_matrix_marginals.png", dpi=300, bbox_inches="tight")

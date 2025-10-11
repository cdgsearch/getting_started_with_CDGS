from typing import List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

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

    def plot_transitions(self, samples: Optional[np.ndarray] = None, title: Optional[str] = None, annotate_valid: bool = False, 
                        num_stds: float = 3.0, n: int = 25) -> plt.Figure:
        """Plot the dataset transitions with Start/End on x-axis and values on y-axis.
        
        Args:
            title: Optional plot title
            annotate_valid: Whether to color-code transitions as green (valid) or red (invalid)
            num_stds: Number of standard deviations to consider as "in-distribution"
            n: Number of samples to plot (default: 25)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot Gaussian distributions
        y_range = (self.data.min() - 0.5, self.data.max() + 0.5)
        y_points = np.linspace(y_range[0], y_range[1], 200)
        
        # Start distributions
        for i, (mean, std) in enumerate(zip(self.start_means, self.start_stds)):
            pdf = norm.pdf(y_points, mean, std) * 0.3
            ax.fill_betweenx(y_points, 0, pdf, alpha=0.3, color=f'C{i}', label=f'Start Mode {i}')
        
        # End distributions  
        for i, (mean, std) in enumerate(zip(self.end_means, self.end_stds)):
            pdf = norm.pdf(y_points, mean, std) * 0.3
            ax.fill_betweenx(y_points, 1, 1 + pdf, alpha=0.3, 
                           color=f'C{i + len(self.start_means)}', label=f'End Mode {i}')
        
        # Plot data points and connections
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(np.where(self.transition_matrix)[0])))
        
        samples = self.data if samples is None else samples
        for i, (mode_x1, mode_x2) in enumerate(zip(*np.where(self.transition_matrix))):
            mask = (self.start_labels == mode_x1) & (self.end_labels == mode_x2)
            if not np.any(mask):
                continue
                
            # Limit to first n samples for this transition
            indices = np.where(mask)[0][:n]
            start_vals, end_vals = self.data[indices][:, 0], self.data[indices][:, 1]
            
            # Scatter points
            ax.scatter([0] * len(start_vals), start_vals, c=[colors[i]], alpha=0.6, s=20,
                      label=f'Transition ({mode_x1}→{mode_x2})')
            ax.scatter([1] * len(end_vals), end_vals, c=[colors[i]], alpha=0.6, s=20)
            
            # Connection lines
            for start_val, end_val in zip(start_vals, end_vals):
                if annotate_valid:
                    is_valid = self._is_valid(start_val, end_val, mode_x1, mode_x2, num_stds)
                    color = 'green' if is_valid else 'red'
                    ax.plot([0, 1], [start_val, end_val], color=color, alpha=0.8, linewidth=1.2)
                else:
                    ax.plot([0, 1], [start_val, end_val], color=colors[i], alpha=0.3, linewidth=0.8)
        
        # Format
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Start', 'End'], fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontsize=16)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig
    
    def _is_valid(self, start_val: float, end_val: float, start_mode: int, end_mode: int, num_stds: float) -> bool:
        """Check if a transition is within expected distributions."""
        start_valid = abs(start_val - self.start_means[start_mode]) <= num_stds * self.start_stds[start_mode]
        end_valid = abs(end_val - self.end_means[end_mode]) <= num_stds * self.end_stds[end_mode]
        return start_valid and end_valid
    
    def compute_accuracy(self, samples: np.ndarray, num_stds: float = 3.0) -> float:
        """Compute accuracy of samples against theoretical distributions.
        
        Args:
            samples: Array of shape (N, 2) with start and end values
            num_stds: Number of standard deviations to consider as "in-distribution"
            
        Returns:
            float: Fraction of samples that are within the expected distributions
        """
        if samples.shape[1] != 2:
            raise ValueError("Samples must have shape (N, 2)")
        
        valid_count = 0
        total_count = len(samples)
        
        for start_val, end_val in samples:
            # Check if sample is valid for any allowed transition
            is_sample_valid = False
            
            for mode_x1, mode_x2 in zip(*np.where(self.transition_matrix)):
                start_valid = abs(start_val - self.start_means[mode_x1]) <= num_stds * self.start_stds[mode_x1]
                end_valid = abs(end_val - self.end_means[mode_x2]) <= num_stds * self.end_stds[mode_x2]
                
                if start_valid and end_valid:
                    is_sample_valid = True
                    break
            
            if is_sample_valid:
                valid_count += 1
        
        return valid_count / total_count

if __name__ == "__main__":
    # Example usage
    transition_matrix = np.array([
        [1, 0],  # Mode 0 of start can only go to mode 0 of end
        [0, 1],  # Mode 1 of start can only go to mode 1 of end
    ], dtype=bool)
    
    dataset = MultiModalDataset(
        num_samples=1000,
        start_means=[0, 2],
        start_stds=[0.2, 0.2],
        end_means=[0, 1],
        end_stds=[0.2, 0.2],
        transition_matrix=transition_matrix
    )
    
    # Plot transitions
    fig = dataset.plot_transitions(title="Dataset Transitions")
    fig.savefig("figures_notebook/custom_transition_matrix_dataset.png", dpi=300, bbox_inches="tight")
    
    # Test accuracy computation
    accuracy = dataset.compute_accuracy(dataset.data)
    print(f"Dataset self-accuracy: {accuracy:.3f}")
    
    # Test with some random samples
    random_samples = np.random.randn(100, 2)
    random_accuracy = dataset.compute_accuracy(random_samples)
    print(f"Random samples accuracy: {random_accuracy:.3f}")
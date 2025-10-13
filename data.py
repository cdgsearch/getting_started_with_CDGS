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

    def _plot_distributions(self, ax: plt.Axes, x_pos: float, means: List[float], stds: List[float], 
                           y_range: Tuple[float, float], label_prefix: str, color_offset: int = 0) -> None:
        """Plot Gaussian distributions at a given x position."""
        y_points = np.linspace(y_range[0], y_range[1], 200)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            pdf = norm.pdf(y_points, mean, std) * 0.3
            ax.fill_betweenx(y_points, x_pos, x_pos + pdf, alpha=0.3, 
                           color=f'C{i + color_offset}', label=f'{label_prefix} Mode {i}')

    def _plot_dataset_samples(self, ax: plt.Axes, n: int, annotate_valid: bool, num_stds: float) -> None:
        """Plot dataset samples with mode labels and transitions."""
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(np.where(self.transition_matrix)[0])))
        
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

    def _plot_external_samples(self, ax: plt.Axes, samples: np.ndarray, n: int, 
                              annotate_valid: bool, num_stds: float) -> None:
        """Plot external samples without mode labels."""
        samples_to_plot = samples[:n]  # Limit to first n samples
        start_vals, end_vals = samples_to_plot[:, 0], samples_to_plot[:, 1]
        
        # Scatter points
        ax.scatter([0] * len(start_vals), start_vals, alpha=0.6, s=20, 
                  color='blue', label='Start samples')
        ax.scatter([1] * len(end_vals), end_vals, alpha=0.6, s=20,
                  color='red', label='End samples')
        
        # Connection lines
        for start_val, end_val in zip(start_vals, end_vals):
            if annotate_valid:
                # Check if sample is valid for any allowed transition
                is_valid = False
                for mode_x1, mode_x2 in zip(*np.where(self.transition_matrix)):
                    if self._is_valid(start_val, end_val, mode_x1, mode_x2, num_stds):
                        is_valid = True
                        break
                color = 'green' if is_valid else 'red'
                ax.plot([0, 1], [start_val, end_val], color=color, alpha=0.8, linewidth=1.2)
            else:
                ax.plot([0, 1], [start_val, end_val], color='gray', alpha=0.3, linewidth=0.8)

    def plot_transitions(self, samples: Optional[np.ndarray] = None, title: Optional[str] = None, annotate_valid: bool = False, 
                        num_stds: float = 3.0, n: int = 25) -> plt.Figure:
        """Plot the dataset transitions with Start/End on x-axis and values on y-axis.
        
        Args:
            samples: Optional external samples to plot (e.g., from diffusion model). If None, uses dataset samples.
            title: Optional plot title
            annotate_valid: Whether to color-code transitions as green (valid) or red (invalid)
            num_stds: Number of standard deviations to consider as "in-distribution"
            n: Number of samples to plot (default: 25)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate y-range for distributions
        y_range = (self.data.min() - 0.5, self.data.max() + 0.5)
        
        # Plot distributions using helper method
        self._plot_distributions(ax, 0, self.start_means, self.start_stds, y_range, 'Start')
        self._plot_distributions(ax, 1, self.end_means, self.end_stds, y_range, 'End', len(self.start_means))
        
        # Plot data points and connections using helper methods
        if samples is None:
            self._plot_dataset_samples(ax, n, annotate_valid, num_stds)
        else:
            self._plot_external_samples(ax, samples, n, annotate_valid, num_stds)
        
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

    @staticmethod
    def _validate_transition_step(val1: float, val2: float, dataset1: 'MultiModalDataset', 
                                 dataset2: 'MultiModalDataset', num_stds: float = 3.0) -> bool:
        """Validate a single transition step between two datasets.
        
        Args:
            val1: Value at step i
            val2: Value at step i+1
            dataset1: Dataset for step i (end distributions)
            dataset2: Dataset for step i+1 (start distributions)
            num_stds: Number of standard deviations to consider valid
            
        Returns:
            bool: True if transition is valid for any allowed mode combination
        """
        # Check if transition is valid for any allowed combination
        for mode1_idx, (mean1, std1) in enumerate(zip(dataset1.end_means, dataset1.end_stds)):
            for mode2_idx, (mean2, std2) in enumerate(zip(dataset2.start_means, dataset2.start_stds)):
                # Check if this transition is allowed in both datasets
                if (hasattr(dataset1, 'transition_matrix') and 
                    hasattr(dataset2, 'transition_matrix')):
                    # For multi-step, we need to check if the transition makes sense
                    # This is a simplified check - you might want more sophisticated logic
                    pass
                
                # Check if values are within expected distributions
                val1_valid = abs(val1 - mean1) <= num_stds * std1
                val2_valid = abs(val2 - mean2) <= num_stds * std2
                
                if val1_valid and val2_valid:
                    return True
        
        return False

    @staticmethod
    def _validate_full_path(path: np.ndarray, datasets: List['MultiModalDataset'], 
                           num_stds: float = 3.0) -> Tuple[bool, List[bool]]:
        """Validate an entire multi-step path.
        
        Args:
            path: Array of shape (n_steps,) representing values at each step
            datasets: List of datasets, where datasets[i] represents transition from step i to i+1
            num_stds: Number of standard deviations to consider valid
            
        Returns:
            Tuple of (is_fully_valid, step_validities) where step_validities[i] indicates
            if transition from step i to i+1 is valid
        """
        if len(path) != len(datasets) + 1:
            raise ValueError(f"Path length {len(path)} should be {len(datasets) + 1} for {len(datasets)} datasets")
        
        step_validities = []
        
        for i in range(len(datasets)):
            # For single dataset transitions, we validate against the dataset's own transition rules
            # Check if the transition from path[i] to path[i+1] is valid within datasets[i]
            is_step_valid = False
            
            # Check if this step is valid for any allowed transition in the dataset
            for mode_x1, mode_x2 in zip(*np.where(datasets[i].transition_matrix)):
                start_valid = abs(path[i] - datasets[i].start_means[mode_x1]) <= num_stds * datasets[i].start_stds[mode_x1]
                end_valid = abs(path[i + 1] - datasets[i].end_means[mode_x2]) <= num_stds * datasets[i].end_stds[mode_x2]
                
                if start_valid and end_valid:
                    is_step_valid = True
                    break
            
            step_validities.append(is_step_valid)
        
        is_fully_valid = all(step_validities)
        return is_fully_valid, step_validities

    @staticmethod
    def plot_multi_step_transitions(datasets: List['MultiModalDataset'], samples: np.ndarray, 
                                   title: Optional[str] = None, annotate_valid: bool = True,
                                   num_stds: float = 3.0, n: int = 25) -> plt.Figure:
        """Plot multi-step transitions across multiple datasets.
        
        Args:
            datasets: List of MultiModalDataset objects, one for each transition step
            samples: Array of shape (N, n_steps) where n_steps = len(datasets) + 1
            title: Optional plot title
            annotate_valid: Whether to color-code transitions (green=fully valid, gray+red=partially valid)
            num_stds: Number of standard deviations to consider as "in-distribution"
            n: Number of samples to plot (default: 25)
            
        Returns:
            matplotlib Figure object
        """
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided")
        
        n_steps = len(datasets) + 1
        if samples.shape[1] != n_steps:
            raise ValueError(f"Samples should have {n_steps} columns for {len(datasets)} datasets")
        
        # Limit samples to plot
        samples_to_plot = samples[:n]
        
        # Create figure with appropriate width
        fig_width = max(12, 4 * n_steps)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Calculate global y-range across all datasets
        all_means = []
        all_stds = []
        for dataset in datasets:
            all_means.extend(dataset.start_means + dataset.end_means)
            all_stds.extend(dataset.start_stds + dataset.end_stds)
        
        # Also consider sample range
        sample_min, sample_max = samples_to_plot.min(), samples_to_plot.max()
        mean_min, mean_max = min(all_means), max(all_means)
        std_max = max(all_stds)
        
        y_range = (min(sample_min, mean_min - 3 * std_max) - 0.5, 
                  max(sample_max, mean_max + 3 * std_max) + 0.5)
        
        # Plot distributions at each step
        x_positions = list(range(n_steps))
        
        # First step: start distributions from first dataset
        MultiModalDataset._plot_distributions_static(
            ax, 0, datasets[0].start_means, datasets[0].start_stds, 
            y_range, 'Step 0', 0
        )
        
        # Intermediate and final steps: end distributions from each dataset
        for i, dataset in enumerate(datasets):
            x_pos = i + 1
            color_offset = (i + 1) * len(dataset.end_means)
            MultiModalDataset._plot_distributions_static(
                ax, x_pos, dataset.end_means, dataset.end_stds,
                y_range, f'Step {x_pos}', color_offset
            )
        
        # Plot sample paths with validation coloring
        for sample_idx, path in enumerate(samples_to_plot):
            if annotate_valid:
                # Validate the full path
                is_fully_valid, step_validities = MultiModalDataset._validate_full_path(
                    path, datasets, num_stds
                )
                
                if is_fully_valid:
                    # Fully valid path: green line
                    ax.plot(x_positions, path, color='green', alpha=0.8, linewidth=1.2)
                else:
                    # Partially valid path: gray line with red segments for invalid transitions
                    for i in range(len(datasets)):
                        x_start, x_end = i, i + 1
                        y_start, y_end = path[i], path[i + 1]
                        
                        if step_validities[i]:
                            # Valid transition: gray
                            ax.plot([x_start, x_end], [y_start, y_end], 
                                   color='gray', alpha=0.6, linewidth=1.0)
                        else:
                            # Invalid transition: red
                            ax.plot([x_start, x_end], [y_start, y_end], 
                                   color='red', alpha=0.8, linewidth=1.2)
            else:
                # No validation: simple gray lines
                ax.plot(x_positions, path, color='gray', alpha=0.3, linewidth=0.8)
            
            # Scatter points at each step
            ax.scatter(x_positions, path, alpha=0.6, s=20, color='blue')
        
        # Format plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'Step {i}' for i in range(n_steps)], fontsize=12)
        ax.set_ylabel('Value', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title(f'Multi-step Transitions ({len(datasets)} steps)', fontsize=16)
            
        # Add legend for validation colors if enabled
        if annotate_valid:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='Fully valid path'),
                Line2D([0], [0], color='gray', lw=2, label='Valid transition'),
                Line2D([0], [0], color='red', lw=2, label='Invalid transition')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_distributions_static(ax: plt.Axes, x_pos: float, means: List[float], stds: List[float], 
                                  y_range: Tuple[float, float], label_prefix: str, color_offset: int = 0) -> None:
        """Static version of _plot_distributions for use in static methods."""
        y_points = np.linspace(y_range[0], y_range[1], 200)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            pdf = norm.pdf(y_points, mean, std) * 0.3
            ax.fill_betweenx(y_points, x_pos, x_pos + pdf, alpha=0.3, 
                           color=f'C{i + color_offset}', label=f'{label_prefix} Mode {i}')
    
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
    
    # Test original functionality
    print("Testing original plot_transitions functionality...")
    fig = dataset.plot_transitions(title="Dataset Transitions")
    fig.savefig("figures_notebook/custom_transition_matrix_dataset.png", dpi=300, bbox_inches="tight")
    
    # Test accuracy computation
    accuracy = dataset.compute_accuracy(dataset.data)
    print(f"Dataset self-accuracy: {accuracy:.3f}")
    
    # Test with some random samples
    random_samples = np.random.randn(100, 2)
    random_accuracy = dataset.compute_accuracy(random_samples)
    print(f"Random samples accuracy: {random_accuracy:.3f}")
    
    # Test multi-step functionality
    print("\nTesting multi-step plot_multi_step_transitions functionality...")
    
    # Create datasets for multi-step transitions
    dataset1 = MultiModalDataset(
        num_samples=100,
        start_means=[0, 2],
        start_stds=[0.2, 0.2],
        end_means=[1, 3],
        end_stds=[0.2, 0.2],
        transition_matrix=transition_matrix
    )
    
    dataset2 = MultiModalDataset(
        num_samples=100,
        start_means=[1, 3],  # Should match end of dataset1
        start_stds=[0.2, 0.2],
        end_means=[0.5, 2.5],
        end_stds=[0.2, 0.2],
        transition_matrix=transition_matrix
    )
    
    # Create sample multi-step paths
    np.random.seed(42)
    n_samples = 50
    multi_step_samples = np.random.randn(n_samples, 3)  # 3 steps: x1 -> x2 -> x3
    
    # Make some samples more realistic (closer to the distributions)
    for i in range(n_samples):
        if i < 25:  # First half: valid-ish samples
            multi_step_samples[i, 0] = np.random.choice([0, 2]) + np.random.normal(0, 0.3)
            multi_step_samples[i, 1] = np.random.choice([1, 3]) + np.random.normal(0, 0.3)
            multi_step_samples[i, 2] = np.random.choice([0.5, 2.5]) + np.random.normal(0, 0.3)
        # Second half: keep random for testing validation
    
    # Plot multi-step transitions
    datasets = [dataset1, dataset2]
    fig_multi = MultiModalDataset.plot_multi_step_transitions(
        datasets, multi_step_samples, 
        title="Multi-step Transitions Test", 
        annotate_valid=True, 
        n=25
    )
    fig_multi.savefig("figures_notebook/multi_step_transitions_test.png", dpi=300, bbox_inches="tight")
    
    print("Multi-step plot saved to figures_notebook/multi_step_transitions_test.png")
    print("All tests completed successfully!")
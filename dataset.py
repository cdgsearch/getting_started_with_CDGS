"""Utilities for creating and visualizing simple multimodal datasets.

This module provides a small Dataset implementation that produces 2D
multimodal synthetic data with configurable mode counts, spacings,
and transition constraints. It also includes helper functions to
create common transition matrices and simple plotting utilities used
in the accompanying notebook.

Notes:
- The generated dataset is deterministic when a `seed` is provided.
- The transition matrix controls which (x1, x2) mode pairs are
    allowed; invalid pairs will not be sampled.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MultiModal2DDataset(Dataset):
    """A lightweight 2D multimodal synthetic dataset.

    The dataset generates `num_samples` points in R^2 by sampling from
    independent mixture components along each axis (x1 and x2). A
    `transition_matrix` can be provided to restrict which combinations
    of (x1 mode, x2 mode) are valid; only valid pairs will be sampled.
    """

    def __init__(self, num_samples, num_modes_x1=1, num_modes_x2=1,
                 mode_spacing_x1=2.0, mode_spacing_x2=2.0,
                 mode_std_x1=0.1, mode_std_x2=0.1,
                 mode_probs_x1=None, mode_probs_x2=None,
                 transition_matrix=None, seed=None):
        """Initialize dataset and pre-sample points.

        See module-level docs for detailed parameter descriptions. This
        method pre-samples `num_samples` points according to the
        provided transition_matrix and mode probabilities.
        """
        self.num_samples = num_samples
        self.num_modes_x1 = num_modes_x1
        self.num_modes_x2 = num_modes_x2
        self.mode_spacing_x1 = mode_spacing_x1
        self.mode_spacing_x2 = mode_spacing_x2
        self.mode_std_x1 = mode_std_x1
        self.mode_std_x2 = mode_std_x2
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Transition matrix for valid mode combinations
        if transition_matrix is None:
            # Default: all combinations are valid
            self.transition_matrix = np.ones((num_modes_x1, num_modes_x2), dtype=bool)
        else:
            self.transition_matrix = np.array(transition_matrix, dtype=bool)
            if self.transition_matrix.shape != (num_modes_x1, num_modes_x2):
                raise ValueError(
                    f"Transition matrix shape {self.transition_matrix.shape} "
                    f"doesn't match expected shape ({num_modes_x1}, {num_modes_x2})"
                )

        # Get valid mode combinations
        valid_combinations = np.where(self.transition_matrix)
        self.valid_mode_pairs = list(zip(valid_combinations[0], valid_combinations[1]))

        if len(self.valid_mode_pairs) == 0:
            raise ValueError("No valid mode combinations specified in transition matrix")

        # Create probabilities for valid combinations
        self.combination_probs = []
        for mode_x1, mode_x2 in self.valid_mode_pairs:
            prob_x1 = self.mode_probs_x1[mode_x1] if mode_probs_x1 is not None else 1.0 / num_modes_x1
            prob_x2 = self.mode_probs_x2[mode_x2] if mode_probs_x2 is not None else 1.0 / num_modes_x2
            self.combination_probs.append(prob_x1 * prob_x2)

        # Normalize combination probabilities
        self.combination_probs = np.array(self.combination_probs)
        self.combination_probs = self.combination_probs / self.combination_probs.sum()

        # Mode centers for x1 and x2
        self.centers_x1 = np.linspace(-((num_modes_x1 - 1) / 2) * mode_spacing_x1,
                                     ((num_modes_x1 - 1) / 2) * mode_spacing_x1,
                                     num_modes_x1)
        self.centers_x2 = np.linspace(-((num_modes_x2 - 1) / 2) * mode_spacing_x2,
                                     ((num_modes_x2 - 1) / 2) * mode_spacing_x2,
                                     num_modes_x2)

        # Probabilities for each mode in x1
        if mode_probs_x1 is None:
            self.mode_probs_x1 = np.ones(num_modes_x1) / num_modes_x1
        else:
            self.mode_probs_x1 = np.array(mode_probs_x1)
            self.mode_probs_x1 = self.mode_probs_x1 / self.mode_probs_x1.sum()

        # Probabilities for each mode in x2
        if mode_probs_x2 is None:
            self.mode_probs_x2 = np.ones(num_modes_x2) / num_modes_x2
        else:
            self.mode_probs_x2 = np.array(mode_probs_x2)
            self.mode_probs_x2 = self.mode_probs_x2 / self.mode_probs_x2.sum()

        # Pre-sample all points
        self.data = []
        self.labels_x1 = []
        self.labels_x2 = []

        for _ in range(num_samples):
            # Sample a valid mode combination
            combo_idx = self.rng.choice(len(self.valid_mode_pairs), p=self.combination_probs)
            mode_idx_x1, mode_idx_x2 = self.valid_mode_pairs[combo_idx]

            # Sample x1 and x2 from selected modes
            x1 = self.rng.normal(loc=self.centers_x1[mode_idx_x1], scale=mode_std_x1)
            x2 = self.rng.normal(loc=self.centers_x2[mode_idx_x2], scale=mode_std_x2)

            self.data.append([x1, x2])
            self.labels_x1.append(mode_idx_x1)
            self.labels_x2.append(mode_idx_x2)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels_x1 = np.array(self.labels_x1, dtype=np.int64)
        self.labels_x2 = np.array(self.labels_x2, dtype=np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Returns data as `(x1, x2)` points with labels `(mode_idx_x1, mode_idx_x2)`
        return torch.tensor(self.data[idx])

    def plot_transition_matrix(self, figsize=(8, 6), title=None):
        """Visualize the transition matrix showing valid mode combinations."""
        plt.figure(figsize=figsize)

        # Create heatmap of transition matrix
        plt.imshow(self.transition_matrix.T, cmap='RdYlBu_r', aspect='auto', origin='lower')
        plt.colorbar(label='Valid Transition (1=Yes, 0=No)')

        # Labels and ticks
        plt.xlabel('X1 Mode Index')
        plt.ylabel('X2 Mode Index')
        plt.title(title or 'Valid Mode Transitions Matrix')

        x1_labels = [f'{i}\n({self.centers_x1[i]:.1f})' for i in range(self.num_modes_x1)]
        x2_labels = [f'{i}\n({self.centers_x2[i]:.1f})' for i in range(self.num_modes_x2)]
        plt.xticks(range(self.num_modes_x1), x1_labels)
        plt.yticks(range(self.num_modes_x2), x2_labels)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("transition_matrix.png")

    def get_valid_combinations_info(self):
        """Return information about valid mode combinations and observed coverage."""
        unique_combinations = set(zip(self.labels_x1, self.labels_x2))

        info = {
            'total_valid_combinations': len(self.valid_mode_pairs),
            'total_possible_combinations': self.num_modes_x1 * self.num_modes_x2,
            'valid_mode_pairs': self.valid_mode_pairs,
            'combination_probabilities': dict(zip(self.valid_mode_pairs, self.combination_probs)),
            'observed_combinations': list(unique_combinations),
            'transition_matrix_shape': self.transition_matrix.shape,
            'coverage_ratio': len(self.valid_mode_pairs) / (self.num_modes_x1 * self.num_modes_x2),
        }

        return info

    def plot_data(self, figsize=(10, 8), alpha=0.6, s=20, title=None):
        """Plot the 2D dataset with different colors for different mode combinations."""
        plt.figure(figsize=figsize)

        unique_combinations = set(zip(self.labels_x1, self.labels_x2))
        n_combinations = len(unique_combinations)
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_combinations))

        for i, (mode_x1, mode_x2) in enumerate(unique_combinations):
            mask = (self.labels_x1 == mode_x1) & (self.labels_x2 == mode_x2)
            data_x1 = self.data[mask][:, 0]
            data_x2 = self.data[mask][:, 1]
            plt.scatter(data_x1, data_x2, c=[colors[i]], alpha=alpha, s=s,
                        label=f'Mode ({mode_x1}, {mode_x2})')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title or f'2D Multimodal Dataset ({self.num_modes_x1}×{self.num_modes_x2} modes)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("2D_multimodal_dataset.png")

    def plot_marginals(self, figsize=(12, 4), bins=50, alpha=0.7, title=None):
        """Plot marginal distributions of X1 and X2."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # X1 marginal
        data_x1 = self.data[:, 0]  # type: ignore
        ax1.hist(data_x1, bins=bins, alpha=alpha, density=True, color='blue', edgecolor='black')
        for center in self.centers_x1:
            ax1.axvline(center, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(self.centers_x1[0], color='red', linestyle='--', alpha=0.7, label='Mode centers')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('Density')
        ax1.set_title(f'X1 Marginal ({self.num_modes_x1} modes)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # X2 marginal
        data_x2 = self.data[:, 1]  # type: ignore
        ax2.hist(data_x2, bins=bins, alpha=alpha, density=True, color='green', edgecolor='black')
        for center in self.centers_x2:
            ax2.axvline(center, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(self.centers_x2[0], color='red', linestyle='--', alpha=0.7, label='Mode centers')
        ax2.set_xlabel('X2')
        ax2.set_ylabel('Density')
        ax2.set_title(f'X2 Marginal ({self.num_modes_x2} modes)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.savefig("2D_marginals.png")


def plot_1d_dataset(dataset, figsize=(10, 6), bins=50, alpha=0.7, title=None):
    """
    Plot utility for 1D multimodal dataset.
    
    Args:
        dataset (MultiModal1DDataset): The 1D dataset to plot.
        figsize (tuple): Figure size (width, height).
        bins (int): Number of histogram bins.
        alpha (float): Histogram transparency.
        title (str): Plot title.
    """
    plt.figure(figsize=figsize)
    
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
    plt.show()


def create_diagonal_transition_matrix(num_modes_x1, num_modes_x2):
    """
    Create a diagonal transition matrix (only allows matching mode indices).
    
    Args:
        num_modes_x1 (int): Number of modes in x1 dimension.
        num_modes_x2 (int): Number of modes in x2 dimension.
        
    Returns:
        np.array: Diagonal transition matrix.
    """
    matrix = np.zeros((num_modes_x1, num_modes_x2), dtype=bool)
    min_modes = min(num_modes_x1, num_modes_x2)
    for i in range(min_modes):
        matrix[i, i] = True
    return matrix


def create_block_diagonal_transition_matrix(num_modes_x1, num_modes_x2, block_size=2):
    """
    Create a block diagonal transition matrix.
    
    Args:
        num_modes_x1 (int): Number of modes in x1 dimension.
        num_modes_x2 (int): Number of modes in x2 dimension.
        block_size (int): Size of each diagonal block.
        
    Returns:
        np.array: Block diagonal transition matrix.
    """
    matrix = np.zeros((num_modes_x1, num_modes_x2), dtype=bool)
    
    for i in range(0, min(num_modes_x1, num_modes_x2), block_size):
        end_i = min(i + block_size, num_modes_x1)
        end_j = min(i + block_size, num_modes_x2)
        matrix[i:end_i, i:end_j] = True
    
    return matrix


def create_sparse_random_transition_matrix(num_modes_x1, num_modes_x2, sparsity=0.3, seed=None):
    """
    Create a sparse random transition matrix.
    
    Args:
        num_modes_x1 (int): Number of modes in x1 dimension.
        num_modes_x2 (int): Number of modes in x2 dimension.
        sparsity (float): Fraction of entries that should be True (0.0 to 1.0).
        seed (int or None): Random seed.
        
    Returns:
        np.array: Sparse random transition matrix.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.random((num_modes_x1, num_modes_x2)) < sparsity
    
    # Ensure at least one valid combination exists
    if not matrix.any():
        matrix[0, 0] = True
    
    return matrix


if __name__ == "__main__":

    transition_matrix = np.array([
        [1, 0],  # Mode 0 of x1 can pair with mode 0,1 of x2
        [0, 1],
    ], dtype=bool)
    
    dataset_custom = MultiModal2DDataset(
        num_samples=5000,
        num_modes_x1=2,
        num_modes_x2=2,
        mode_spacing_x1=2.0,
        mode_spacing_x2=1.5,
        mode_std_x1=0.2,
        mode_std_x2=0.1,
        transition_matrix=transition_matrix
    )
    print("Custom transition matrix:")
    print(transition_matrix.astype(int))
    print("Valid combinations:", dataset_custom.get_valid_combinations_info()['valid_mode_pairs'])
    
    # Plot the custom dataset
    dataset_custom.plot_data(title="Custom Transition Matrix Dataset")
    dataset_custom.plot_transition_matrix()
    dataset_custom.plot_marginals(title="Custom Transitions - Marginal Distributions")

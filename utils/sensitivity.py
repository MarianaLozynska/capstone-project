"""
Sensitivity analysis utilities for capstone project
"""

import numpy as np
import matplotlib.pyplot as plt
from .bayesian_optimization import fit_gp


def sensitivity_analysis(func_id, X_sample, y_sample, n_points=50, gp=None):
    """
    Perform sensitivity analysis by varying each dimension independently

    Args:
        func_id: Function ID
        X_sample: Observed inputs
        y_sample: Observed outputs
        n_points: Number of points to sample along each dimension
        gp: Pre-fitted GP model (if None, fits a new one)
    """
    dim = X_sample.shape[1]

    # Reuse existing GP or fit a new one
    if gp is None:
        gp = fit_gp(X_sample, y_sample)

    # Use best point as baseline
    best_idx = np.argmax(y_sample)
    baseline = X_sample[best_idx].copy()

    # Create figure
    n_cols = min(4, dim)
    n_rows = (dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print(f"\n{'='*70}")
    print(f"Function {func_id} ({dim}D) - Sensitivity Analysis")
    print(f"{'='*70}")
    print(f"Baseline (best point): {baseline}")
    print(f"Baseline output: {y_sample[best_idx]:.6f}\n")

    # Analyze each dimension
    for d in range(dim):
        # Vary dimension d from 0 to 1
        test_points = np.tile(baseline, (n_points, 1))
        test_points[:, d] = np.linspace(0, 1, n_points)

        # Get GP predictions
        mu, sigma = gp.predict(test_points, return_std=True)

        # Plot
        ax = axes[d]
        ax.plot(test_points[:, d], mu, 'b-', linewidth=2, label='GP mean')
        ax.fill_between(test_points[:, d], mu - sigma, mu + sigma, alpha=0.3, label='±1σ')
        ax.axvline(baseline[d], color='r', linestyle='--', linewidth=2, label=f'Current: {baseline[d]:.3f}')
        ax.axhline(y_sample[best_idx], color='g', linestyle=':', linewidth=1, alpha=0.5)

        # Mark observed points in this dimension
        obs_in_dim = X_sample[:, d]
        ax.scatter(obs_in_dim, y_sample, c='orange', s=50, alpha=0.6, zorder=5)

        ax.set_xlabel(f'Dimension {d+1}', fontsize=11)
        ax.set_ylabel('Predicted Output', fontsize=11)
        ax.set_title(f'Dim {d+1} Sensitivity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Calculate gradient (sensitivity)
        gradient = np.gradient(mu)
        max_gradient = np.max(np.abs(gradient))
        print(f"Dimension {d+1}: Max gradient = {max_gradient:.4f} {'(steep - important)' if max_gradient > 1 else '(flat - less important)'}")

    # Hide unused subplots
    for idx in range(dim, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    print(f"{'='*70}\n")

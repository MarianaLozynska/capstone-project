"""
Visualization utilities for capstone project
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from bayesian_optimization import fit_gp


def visualize_gp_2d(func_id, X_sample, y_sample, next_point, resolution=100):
    """
    Visualize GP predictions and next recommended point for 2D functions

    Args:
        func_id: Function number (1 or 2)
        X_sample: Observed input points (n_samples, 2)
        y_sample: Observed output values (n_samples,)
        next_point: Next recommended point (2,)
        resolution: Grid resolution for contour plots
    """
    if X_sample.shape[1] != 2:
        print(f"Function {func_id} is {X_sample.shape[1]}D - skipping 2D visualization")
        return

    # Fit Gaussian Process
    gp = fit_gp(X_sample, y_sample)

    # Create mesh grid
    x1 = np.linspace(0, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # Predict on grid
    y_pred, sigma = gp.predict(X_grid, return_std=True)
    y_pred = y_pred.reshape(X1.shape)
    sigma = sigma.reshape(X1.shape)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Observed data
    ax = axes[0]
    scatter = ax.scatter(X_sample[:, 0], X_sample[:, 1], c=y_sample, s=100, cmap='viridis',
                        edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(next_point[0], next_point[1], c='red', s=400, marker='*',
              edgecolors='white', linewidths=3, label='Next point', zorder=10)
    plt.colorbar(scatter, ax=ax, label='Output value')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(f'Function {func_id}: Observed Data\n({len(X_sample)} points)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Plot 2: GP mean prediction
    ax = axes[1]
    contour = ax.contourf(X1, X2, y_pred, levels=20, cmap='viridis')
    ax.scatter(X_sample[:, 0], X_sample[:, 1], c='white', s=80,
              edgecolors='black', linewidths=2, label='Observed', zorder=5)
    ax.scatter(next_point[0], next_point[1], c='red', s=400, marker='*',
              edgecolors='white', linewidths=3, label='Next point', zorder=10)
    plt.colorbar(contour, ax=ax, label='Predicted output')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(f'Function {func_id}: GP Mean Prediction\n(GP model of function)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Plot 3: GP uncertainty
    ax = axes[2]
    contour = ax.contourf(X1, X2, sigma, levels=20, cmap='RdYlGn_r')
    ax.scatter(X_sample[:, 0], X_sample[:, 1], c='blue', s=80,
              edgecolors='white', linewidths=2, label='Observed', zorder=5)
    ax.scatter(next_point[0], next_point[1], c='red', s=400, marker='*',
              edgecolors='white', linewidths=3, label='Next point', zorder=10)
    plt.colorbar(contour, ax=ax, label='Uncertainty (σ)')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(f'Function {func_id}: GP Uncertainty\n(Where GP is uncertain)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"Function {func_id} Visualization Summary")
    print(f"{'='*70}")
    print(f"Current best: {np.max(y_sample):.6f} at {X_sample[np.argmax(y_sample)]}")
    print(f"Next point: {next_point}")
    pred_mean = gp.predict(next_point.reshape(1, -1))[0]
    pred_std = gp.predict(next_point.reshape(1, -1), return_std=True)[1][0]
    print(f"GP prediction at next point: {pred_mean:.6f} ± {pred_std:.6f}")
    print(f"{'='*70}\n")


def visualize_high_dim(func_id, X_sample, y_sample, next_point):
    """
    Visualize higher-dimensional functions using PCA and GP validation

    Args:
        func_id: Function number (3-8)
        X_sample: Observed input points (n_samples, n_dims)
        y_sample: Observed output values (n_samples,)
        next_point: Next recommended point (n_dims,)
    """
    dim = X_sample.shape[1]

    # Fit Gaussian Process
    gp = fit_gp(X_sample, y_sample)

    # Get GP predictions for all points
    y_pred = gp.predict(X_sample)
    r2 = r2_score(y_sample, y_pred)

    # PCA projection to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    next_point_pca = pca.transform(next_point.reshape(1, -1))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: PCA projection
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, s=100, cmap='viridis',
                        edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(next_point_pca[0, 0], next_point_pca[0, 1], c='red', s=400, marker='*',
              edgecolors='white', linewidths=3, label='Next point', zorder=10)
    plt.colorbar(scatter, ax=ax, label='Output value')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'Function {func_id} ({dim}D): PCA Projection\n' +
                f'Explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Plot 2: Predicted vs Actual
    ax = axes[1]
    ax.scatter(y_sample, y_pred, s=100, alpha=0.6, edgecolors='black', linewidths=1)

    # Perfect prediction line
    min_val = min(y_sample.min(), y_pred.min())
    max_val = max(y_sample.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual Output', fontsize=12)
    ax.set_ylabel('GP Predicted Output', fontsize=12)
    ax.set_title(f'Function {func_id}: GP Model Validation\nR² = {r2:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"Function {func_id} ({dim}D) Summary")
    print(f"{'='*70}")
    print(f"Data points: {len(X_sample)}")
    print(f"Current best: {np.max(y_sample):.6f}")
    fit_quality = 'Excellent fit' if r2 > 0.9 else 'Good fit' if r2 > 0.7 else 'Needs more data'
    print(f"GP R² score: {r2:.4f} ({fit_quality})")
    print(f"PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    next_pred = gp.predict(next_point.reshape(1, -1))[0]
    next_std = gp.predict(next_point.reshape(1, -1), return_std=True)[1][0]
    print(f"Next point prediction: {next_pred:.6f} ± {next_std:.6f}")
    print(f"{'='*70}\n")


def visualize_all_functions(inputs, outputs, recommendations):
    """
    Generate visualizations for all functions

    Args:
        inputs: Dict of input arrays {func_id: array}
        outputs: Dict of output arrays {func_id: array}
        recommendations: Dict of recommended points {func_id: array}
    """
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # 2D functions
    print("\n2D Functions (1-2): Full GP landscape visualization")
    for func_id in [1, 2]:
        visualize_gp_2d(func_id, inputs[func_id], outputs[func_id],
                       recommendations[func_id], resolution=100)

    # Higher-dimensional functions
    print("\nHigher-Dimensional Functions (3-8): PCA projection + GP validation")
    for func_id in range(3, 9):
        visualize_high_dim(func_id, inputs[func_id], outputs[func_id],
                          recommendations[func_id])

    print("\n" + "="*70)
    print("Visualization Summary:")
    print("="*70)
    print("Functions 1-2: Mean, uncertainty, and data")
    print("Functions 3-8: PCA projection + model validation")
    print("Red star = Next recommended point")
    print("="*70)

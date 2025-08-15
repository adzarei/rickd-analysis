"""This module contains helper functions for timeseries data."""

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import warnings
import numpy as np
import dtw as dtw
from typing import Optional, List, Tuple

def get_group_curves(ts, side: str = "L", num_points: int = 101, data_key: Optional[str] = None):
    """Get the group of joint angle curves for a given session id and joint name."""
    ts_normalized_on_stance = ktk.cycles.time_normalize(
        ts, event_name1=f"{side}_TD", event_name2=f"{side}_TO", n_points=num_points
    )

    data = ktk.cycles.stack(ts_normalized_on_stance, n_points=num_points)
    if data_key is None:
        return data[next(iter(data))]  # Return first key
    else:
        return data[data_key]


def plot_group_curves(group_of_curves, group_name: str, data_description: str = "Joint Angle", Y_axis_unit: str = "Degs", X_axis_unit: str = "% of Stance Phase"):
    """Plot the group of joint angle curves."""
    n_cycles = group_of_curves.shape[0]

    plt.figure(figsize=(10, 10))
    for i, axis in enumerate(["X", "Y", "Z"]):
        ax = plt.subplot(2,2,i + 1)
        for i_cycle in range(n_cycles):
            plt.plot(group_of_curves[i_cycle][:,i])
        plt.title(f"{group_name} - {data_description} {axis} axis - N={n_cycles}")
        plt.xlabel(X_axis_unit)
        plt.ylabel(Y_axis_unit)


def plot_single_curves(single_curve, title: str, Y_axis_unit: str = "Degs", X_axis_unit: str = "% of Stance Phase"):
    """Plot a single curve of joint angle."""
    plt.figure(figsize=(10, 10))
    shape = single_curve.shape

    n_cycles = 1
    # When more than one dimension, we have a group of curves.
    if len(shape) > 1:
        n_cycles = single_curve.shape[0]
        for i_cycle in range(n_cycles):
            plt.plot(single_curve[i_cycle])
    else:
         plt.plot(single_curve)
    plt.title(title)
    plt.xlabel(X_axis_unit)
    plt.ylabel(Y_axis_unit)


def apply_continuous_curve_registration(
            curves: np.ndarray,
            max_iter: int = 10000,
            tol: float = 1e-4,
            plot_dtw: bool = False,
            plot_rate: int = 10,
            plot_type: str = "twoway",
            step_pattern: dtw.StepPattern = dtw.rabinerJuangStepPattern(6, "c")
        ) -> np.ndarray:
    """
    Apply continuous curve registration to a group of curves and return the mean curve.

    NOTE: Curves are iteratively registered using DTW to find the best alignment that respects the carasteristics of the curves.

    Args:
        curves: Array of shape (n_cycles, n_points)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        plot_dtw: Whether to plot the output alignment of the DTW algorithm.
        plot_rate: Plot every plot_rate iterations
        plot_type: Type of plot to use
        step_pattern: Step pattern to use for the alignment
    """
    registered_curves = curves.copy()

    for iteration in range(max_iter):
        if iteration == max_iter - 1:
            warnings.warn("Maximum number of iterations reached. The registration may not have converged.")

        # Compute mean template
        mean_curve = registered_curves.mean(axis=0)
        new_registered = []

        for curve in curves:
            alignment = dtw.dtw(curve, mean_curve, keep_internals=True, step_pattern=step_pattern)

            path_x = alignment.index1  # indices into original curve
            path_y = alignment.index2  # indices into mean curve

            # Create aligned curve
            aligned = np.zeros(len(mean_curve))
            counts = np.zeros(len(mean_curve))

            # Map original curve values to aligned positions
            for i, j in zip(path_x, path_y):
                aligned[j] += curve[i]
                counts[j] += 1

            # Average multiple mappings to same position
            aligned = aligned / np.maximum(counts, 1)
            new_registered.append(aligned)

        registered_curves = np.array(new_registered)

        if plot_dtw and iteration % plot_rate == 0:
            ax = alignment.plot(type=plot_type)
            ax.set_title(f"Iteration {iteration}")

        # Check convergence
        if iteration > 0:
            change = np.mean(np.abs(registered_curves - prev_registered))
            if change < tol:
                print(f"Convergence after {iteration} iterations!")
                break
        prev_registered = registered_curves.copy()
        

    return registered_curves.mean(axis=0)


def plot_compare_features(
    X_ts: np.ndarray, 
    y: np.ndarray, 
    channels: List[str], 
    feature_1_name: str = "L_hip_angle_X",
    feature_2_name: str = "R_knee_angle_Y",
    sample_injured_idx: Optional[int] = None,
    sample_healthy_idx: Optional[int] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot time series data for two samples using two features for comaprison between injury status.
    
    Args:
        X_ts: Time series data of shape (N, T, C) where N=samples, T=timesteps, C=channels
        y: Binary labels (0=healthy, 1=injured) 
        channels: List of channel/feature names corresponding to the C dimension
        feature_1_name: Name of the first feature to plot (must be in channels list)
        feature_2_name: Name of the second feature to plot (must be in channels list)
        sample_injured_idx: Specific injured sample index to plot (if None, uses first injured sample)
        sample_healthy_idx: Specific healthy sample index to plot (if None, uses first healthy sample)
        figsize: Figure size as (width, height)
    """
    
    if feature_1_name not in channels:
        raise ValueError(f"Feature '{feature_1_name}' not found in channels list")
    if feature_2_name not in channels:
        raise ValueError(f"Feature '{feature_2_name}' not found in channels list")
    
    injured_indices = np.where(y == 1)[0]
    healthy_indices = np.where(y == 0)[0]
    
    if len(injured_indices) == 0:
        raise IndexError("No injured samples found in the dataset")
    if len(healthy_indices) == 0:
        raise IndexError("No healthy samples found in the dataset")
    
    # Use provided indices or default to first sample of each class
    sample_injured = sample_injured_idx if sample_injured_idx is not None else injured_indices[0]
    sample_healthy = sample_healthy_idx if sample_healthy_idx is not None else healthy_indices[0]
    
    feature_1_idx = channels.index(feature_1_name)
    feature_2_idx = channels.index(feature_2_name)
    sample_1_feat_1 = X_ts[sample_injured, :, feature_1_idx]
    sample_1_feat_2 = X_ts[sample_injured, :, feature_2_idx]
    sample_2_feat_1 = X_ts[sample_healthy, :, feature_1_idx]
    sample_2_feat_2 = X_ts[sample_healthy, :, feature_2_idx]
    
    time_steps = np.arange(X_ts.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Two Samples, Two Features', fontsize=16)
    
    # Plot Sample 1 (Injured) - Feature 1
    axes[0, 0].plot(time_steps, sample_1_feat_1, 'b-', linewidth=2)
    axes[0, 0].set_title(f'Sample {sample_injured} (Injured) - {feature_1_name}')
    axes[0, 0].set_xlabel('% of Stance Phase')
    axes[0, 0].set_ylabel('Scaled Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Sample 1 (Injured) - Feature 2
    axes[0, 1].plot(time_steps, sample_1_feat_2, 'r-', linewidth=2)
    axes[0, 1].set_title(f'Sample {sample_injured} (Injured) - {feature_2_name}')
    axes[0, 1].set_xlabel('% of Stance Phase')
    axes[0, 1].set_ylabel('Scaled Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Sample 2 (Healthy) - Feature 1
    axes[1, 0].plot(time_steps, sample_2_feat_1, 'b--', linewidth=2)
    axes[1, 0].set_title(f'Sample {sample_healthy} (Healthy) - {feature_1_name}')
    axes[1, 0].set_xlabel('% of Stance Phase')
    axes[1, 0].set_ylabel('Scaled Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Sample 2 (Healthy) - Feature 2
    axes[1, 1].plot(time_steps, sample_2_feat_2, 'r--', linewidth=2)
    axes[1, 1].set_title(f'Sample {sample_healthy} (Healthy) - {feature_2_name}')
    axes[1, 1].set_xlabel('% of Stance Phase')
    axes[1, 1].set_ylabel('Scaled Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_all_features_overlay(
    X_ts: np.ndarray, 
    channels: List[str], 
    alpha: float = 0.2,
    figsize: tuple = (10, 5),
    max_features: Optional[int] = None,
    features: Optional[List[str]] = None,
    plots_per_row: int = 1,
    x_label: str = "% of Stance Phase",
    y_label: str = "Scaled Value",
    n_samples: Optional[int] = None
) -> None:
    """
    Plot selected time series features with all samples overlaid.

    Args:
        X_ts: Time series data of shape (N, T, C) where N=samples, T=timesteps, C=channels
        channels: List of channel/feature names corresponding to the C dimension
        alpha: Transparency level for the overlaid lines (0.0 = transparent, 1.0 = opaque)
        figsize: Figure size as (width, height) for each individual plot
        max_features: Maximum number of features to plot (if None, plots all features)
        features: List of feature names to plot (if None, plots all or up to max_features)
        plots_per_row: Number of plots per row in the figure grid
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        n_samples: Number of samples to plot (if None, plots all samples)
    """

    time_steps = np.arange(X_ts.shape[1])  # (T,)

    if features is not None:
        # Only plot the features specified in the list, in the order given
        feature_indices = [channels.index(f) for f in features if f in channels]
    else:
        # Plot all features or up to max_features
        num_features = X_ts.shape[2]
        n_plot = min(num_features, max_features) if max_features is not None else num_features
        feature_indices = list(range(n_plot))

    n_features = len(feature_indices)
    n_cols = plots_per_row
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False
    )

    if n_samples is not None:
        X_ts = X_ts[:n_samples]

    for idx, feature_idx in enumerate(feature_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        feature_name = channels[feature_idx]

        # Plot all samples for this feature
        for sample_idx in range(X_ts.shape[0]):
            ax.plot(time_steps, X_ts[sample_idx, :, feature_idx], alpha=alpha)

        ax.set_title(f"Feature: {feature_name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the z-score scaler to the data.
    
    Args:
        X: Time series data of shape (N, T, C) where N=samples, T=timesteps, C=channels

    Returns:
        mu: Mean of the data
        sd: Standard deviation of the data
    """
    mu = X.mean(axis=(0,1), keepdims=True)
    sd = X.std(axis=(0,1), keepdims=True)
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply Z-Score normalization to the data.

    NOTE: add 1e-8 to avoid division by zero.
    
    Args:
        X: Time series data of shape (N, T, C) where N=samples, T=timesteps, C=channels
        mu: Mean of the data
        sd: Standard deviation of the data

    Returns:
        Z-scored data of shape (N, T, C)
    """
    return (X - mu) / (sd + 1e-8)


def create_linear_phase_channel(T: int = 101, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create a linearphase channel.
    p_t = t / (T-1), t=0..T-1
    
    Args:
        T: int

    Returns:
        phase: (T,1)
    """
    return np.linspace(0.0, 1.0, T, dtype=dtype)[None, :, None] # (1, T, 1)


def split_unilateral(sessions: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Split a session/trial into two: left side and right side and add a phase channel.

    NOTE: We add a linear phase channel to give the model a sense of the time of the stance phase.
    
    Args:
        sessions: (N, T, C)
        channels: (C,)

    Returns:
        left: (N, T, C)
        right: (N, T, C)
    """
    N, T, C = sessions.shape
    left_idx  = [i for i, n in enumerate(channels) if n.startswith("L_")]
    right_idx = [i for i, n in enumerate(channels) if n.startswith("R_")]
    if not len(left_idx) > 0 or not len(right_idx) > 0:
        raise ValueError("No left or right channels found in the channels list")
    
    left  = sessions[:, :, left_idx]   # (N, T, C/2)
    right = sessions[:, :, right_idx]  # (N, T, C/2)

    p = create_linear_phase_channel(T, dtype=sessions.dtype)  # (T,1)
    p = np.broadcast_to(p, (N, T, 1))
    left  = np.concatenate([left,  p], axis=2)    # (N, T, C/2 + 1)
    right = np.concatenate([right, p], axis=2)    # (N, T, C/2 + 1)
    return left, right


def bilateral_to_unilateral(sessions: np.ndarray, y: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert bilateral data to unilateral data.

    Left and right channels are split and a phase channel is added.

    NOTE: We add a linear phase channel to give the model a sense of the time of the stance phase.
    
    Args:
        sessions: (N, T, C)
        channels: (C,)

    Returns:
        unilateral: (2N, T, C/2 + 1)
        y: (2N,)
    """
    left, right = split_unilateral(sessions, channels)
    y = np.concatenate([y, y], axis=0)
    return np.concatenate([left, right], axis=0), y

"""This module contains functions for evaluating the performance of the model."""

from typing import Tuple, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, f1_score
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap


def pick_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "f1"
) -> Tuple[float, dict]:
    """
    Pick the threshold that maximizes the F1-Score or macro F1-Score.

    NOTE: Normally used on the validation set.

    Args:
        y_true: (N,) True labels
        y_score: (N,) Predicted probabilities
        method: "f1" (default) for standard F1, "macro" for macro F1, balanced for balanced accuracy

    Returns:
        threshold: float
        stats: dict
            For method="f1":
                best_f1: float
                P: float
                R: float
            For method="macro":
                macro_f1: float
            For method="balanced":
                balanced_accuracy: float
    """
    if method == "f1":
        p, r, th = precision_recall_curve(y_true, y_score)
        f1 = 2 * p * r / (p + r + 1e-12)
        th = np.append(th, th[-1])  # align thresholds to same length
        i = np.nanargmax(f1)
        return float(th[i]), {"best_f1": float(f1[i]), "P": float(p[i]), "R": float(r[i])}
    elif method == "macro":
        ths = np.unique(np.concatenate([[0.0], np.sort(y_score), [1.0]]))
        best = {"thr": 0.5, "macro_f1": -1}
        for t in ths:
            y_hat = (y_score >= t).astype(int)
            f1_pos = f1_score(y_true, y_hat, pos_label=1, zero_division=0)
            f1_neg = f1_score(1 - y_true, 1 - y_hat, pos_label=1, zero_division=0)
            macro = 0.5 * (f1_pos + f1_neg)
            if macro > best["macro_f1"]:
                best = {"thr": float(t), "macro_f1": float(macro)}
        return best["thr"], {"macro_f1": best["macro_f1"]}
    elif method == "balanced":
        ths = np.unique(np.concatenate([[0.0], np.sort(y_score), [1.0]]))
        best = {"thr": 0.5, "balanced_accuracy": -1}
        for t in ths:
            y_hat = (y_score >= t).astype(int)
            balanced_accuracy = balanced_accuracy_score(y_true, y_hat)
            if balanced_accuracy > best["balanced_accuracy"]:
                best = {"thr": float(t), "balanced_accuracy": float(balanced_accuracy)}
        return best["thr"], {"balanced_accuracy": best["balanced_accuracy"]}
    else:
        raise ValueError(f"Unknown method: {method}")


def get_unilateral_feature_names(channels: List[str], side: str = "L") -> List[str]:
    """Get feature names for unilateral data.
    
    Args:
        channels: List of bilateral channel names
        side: 'L' for left, 'R' for right
        
    Returns:
        List of unilateral feature names including phase channel
    """
    side_channels = [ch for ch in channels if ch.startswith(f"{side}_")]
    # Remove the side prefix for cleaner names
    clean_names = [ch[2:] for ch in side_channels]  # Remove "L_" or "R_"
    # Add phase channel
    clean_names.append("phase")
    return clean_names


def compute_timeseries_saliency(model: tf.keras.Model,
                               ts_input: np.ndarray,
                               meta_input: Optional[np.ndarray] = None,
                               target_class: Optional[int] = None,
                               method: str = "vanilla") -> np.ndarray:
    """Compute saliency maps for timeseries data with optional metadata.
    
    Works with both single-input (timeseries only) and multi-input (timeseries + metadata) models.
    
    Args:
        model: Trained Keras model
        ts_input: Timeseries input array of shape (batch_size, time_steps, features)
        meta_input: Optional metadata input array of shape (batch_size, meta_features)
        target_class: Class to compute gradients for (0 or 1), if None uses predicted class
        method: Saliency method ('vanilla', 'integrated', 'grad_x_input')
    
    Returns:
        saliency_map: Gradient-based saliency map for timeseries input
    """
    # Determine if model has multiple inputs
    has_metadata = meta_input is not None and len(model.inputs) > 1
    
    if method == "vanilla":
        return _compute_vanilla_timeseries_saliency(model, ts_input, meta_input, target_class, has_metadata)
    elif method == "integrated":
        return _compute_integrated_timeseries_saliency(model, ts_input, meta_input, target_class, has_metadata)
    elif method == "grad_x_input":
        vanilla_grads = _compute_vanilla_timeseries_saliency(model, ts_input, meta_input, target_class, has_metadata)
        return vanilla_grads * ts_input
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'vanilla', 'integrated', 'grad_x_input'")


def _compute_vanilla_timeseries_saliency(model: tf.keras.Model,
                                        ts_input: np.ndarray,
                                        meta_input: Optional[np.ndarray],
                                        target_class: Optional[int],
                                        has_metadata: bool) -> np.ndarray:
    """Internal function for vanilla gradient computation."""
    # Convert to tensors
    ts_tensor = tf.Variable(ts_input.astype(np.float32), dtype=tf.float32)
    
    if has_metadata:
        meta_tensor = tf.constant(meta_input.astype(np.float32), dtype=tf.float32)
        model_inputs = [ts_tensor, meta_tensor]
    else:
        model_inputs = ts_tensor
    
    with tf.GradientTape() as tape:
        tape.watch(ts_tensor)
        predictions = model(model_inputs)
        
        if target_class is None:
            # Use predicted class for each sample
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class case
                target_scores = tf.reduce_max(predictions, axis=1)
            else:
                # Binary case - use the sigmoid output directly
                target_scores = tf.squeeze(predictions)
        else:
            # Use specified class
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class case
                target_scores = predictions[:, target_class]
            else:
                # Binary case
                if target_class == 1:
                    target_scores = tf.squeeze(predictions)
                else:
                    target_scores = 1 - tf.squeeze(predictions)
    
    # Compute gradients
    gradients = tape.gradient(target_scores, ts_tensor)
    return gradients.numpy()


def _compute_integrated_timeseries_saliency(model: tf.keras.Model,
                                           ts_input: np.ndarray,
                                           meta_input: Optional[np.ndarray],
                                           target_class: Optional[int],
                                           has_metadata: bool,
                                           m_steps: int = 50) -> np.ndarray:
    """Internal function for integrated gradient computation."""
    # Use zero baseline
    baseline = np.zeros_like(ts_input)
    
    # Convert to tensors
    ts_input_tf = tf.convert_to_tensor(ts_input.astype(np.float32), dtype=tf.float32)
    baseline_tf = tf.convert_to_tensor(baseline.astype(np.float32), dtype=tf.float32)
    
    if has_metadata:
        meta_tensor = tf.constant(meta_input.astype(np.float32), dtype=tf.float32)
    
    # Generate alphas
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    
    # Initialize integrated gradients
    integrated_grads = tf.zeros_like(ts_input_tf)
    
    for alpha in alphas:
        interpolated = baseline_tf + alpha * (ts_input_tf - baseline_tf)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            
            if has_metadata:
                model_inputs = [interpolated, meta_tensor]
            else:
                model_inputs = interpolated
                
            predictions = model(model_inputs)
            
            if target_class is None:
                # Use predicted class
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    target_scores = tf.reduce_max(predictions, axis=1)
                else:
                    target_scores = tf.squeeze(predictions)
            else:
                # Use specified class
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    target_scores = predictions[:, target_class]
                else:
                    if target_class == 1:
                        target_scores = tf.squeeze(predictions)
                    else:
                        target_scores = 1 - tf.squeeze(predictions)
        
        grads = tape.gradient(target_scores, interpolated)
        integrated_grads += grads / m_steps
    
    # Scale by the input difference
    integrated_grads *= (ts_input_tf - baseline_tf)
    
    return integrated_grads.numpy()


def plot_timeseries_saliency(saliency_map: np.ndarray,
                            channels: List[str],
                            sample_idx: int = 0,
                            top_k: int = 10,
                            title: str = "Timeseries Saliency Analysis",
                            figsize: tuple = (15, 10),
                            save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Plot saliency maps for timeseries data.
    
    Args:
        saliency_map: Saliency map array of shape (batch_size, time_steps, features)
        channels: List of channel/feature names
        sample_idx: Sample index to plot
        top_k: Number of top channels to highlight
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot (optional)
    
    Returns:
        top_channels_idx: Indices of top-k most important channels
        channel_importance: Importance score for each channel
    """
    if sample_idx >= len(saliency_map):
        raise ValueError(f"sample_idx {sample_idx} out of range for saliency_map with {len(saliency_map)} samples")
    
    saliency = np.abs(saliency_map[sample_idx])  # Shape: (time_steps, features)
    
    # Calculate average importance per channel
    channel_importance = np.mean(saliency, axis=0)
    
    # Get top-k most important channels
    top_channels_idx = np.argsort(channel_importance)[-top_k:]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f"{title} - Sample {sample_idx}", fontsize=16, fontweight='bold')
    
    # Plot 1: Heatmap of all channels
    im1 = axes[0].imshow(saliency.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0].set_title('Saliency Map - All Channels')
    axes[0].set_xlabel('Time Steps (% of Stance Phase)')
    axes[0].set_ylabel('Channels')
    
    # Set y-ticks for channels (every 5th channel to avoid crowding)
    step = max(1, len(channels) // 10)  # Show max 10 labels
    tick_positions = range(0, len(channels), step)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([channels[i] for i in tick_positions], fontsize=8)
    
    plt.colorbar(im1, ax=axes[0], label='Saliency Score')
    
    # Plot 2: Top-k important channels over time
    colors = plt.cm.tab10(np.linspace(0, 1, min(top_k, 10)))
    
    for i, ch_idx in enumerate(top_channels_idx):
        color = colors[i % len(colors)]
        axes[1].plot(saliency[:, ch_idx], label=channels[ch_idx], 
                    linewidth=2, color=color)
    
    axes[1].set_title(f'Top {top_k} Most Important Channels Over Time')
    axes[1].set_xlabel('Time Steps (% of Stance Phase)')
    axes[1].set_ylabel('Saliency Score')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return top_channels_idx, channel_importance


def analyze_sample_saliency(model: tf.keras.Model,
                           X_ts: np.ndarray,
                           y: np.ndarray,
                           channels: List[str],
                           X_meta: Optional[np.ndarray] = None,
                           injured_idx: Optional[int] = None,
                           healthy_idx: Optional[int] = None,
                           method: str = "integrated",
                           top_k: int = 10,
                           save_dir: Optional[str] = None) -> dict:
    """Analyze saliency for specific injured and healthy samples.
    
    Args:
        model: Trained Keras model
        X_ts: Timeseries test data
        y: Labels
        channels: Feature names
        X_meta: Optional metadata
        injured_idx: Index of injured sample to analyze (if None, uses first injured)
        healthy_idx: Index of healthy sample to analyze (if None, uses first healthy)
        method: Saliency method to use
        top_k: Number of top features to highlight
        save_dir: Directory to save plots
    
    Returns:
        Dictionary with analysis results
    """
    # Find sample indices if not provided
    if injured_idx is None:
        injured_indices = np.where(y == 1)[0]
        if len(injured_indices) == 0:
            raise ValueError("No injured samples found")
        injured_idx = injured_indices[0]
    
    if healthy_idx is None:
        healthy_indices = np.where(y == 0)[0]
        if len(healthy_indices) == 0:
            raise ValueError("No healthy samples found")
        healthy_idx = healthy_indices[0]
    
    print(f"Analyzing injured sample (index {injured_idx}) and healthy sample (index {healthy_idx})")
    
    # Get model predictions
    if X_meta is not None:
        pred_injured = model.predict([X_ts[injured_idx:injured_idx+1], X_meta[injured_idx:injured_idx+1]])[0, 0]
        pred_healthy = model.predict([X_ts[healthy_idx:healthy_idx+1], X_meta[healthy_idx:healthy_idx+1]])[0, 0]
    else:
        pred_injured = model.predict(X_ts[injured_idx:injured_idx+1])[0, 0]
        pred_healthy = model.predict(X_ts[healthy_idx:healthy_idx+1])[0, 0]
    
    print(f"Model predictions:")
    print(f"  Injured sample (true=1): {pred_injured:.4f}")
    print(f"  Healthy sample (true=0): {pred_healthy:.4f}")
    
    # Compute saliency for both samples
    injured_saliency = compute_timeseries_saliency(
        model, X_ts[injured_idx:injured_idx+1], 
        X_meta[injured_idx:injured_idx+1] if X_meta is not None else None,
        target_class=None, method=method
    )
    
    healthy_saliency = compute_timeseries_saliency(
        model, X_ts[healthy_idx:healthy_idx+1],
        X_meta[healthy_idx:healthy_idx+1] if X_meta is not None else None, 
        target_class=None, method=method
    )
    
    # Plot saliency for injured sample
    print("\nPlotting saliency for injured sample...")
    injured_top_idx, injured_importance = plot_timeseries_saliency(
        injured_saliency, channels, sample_idx=0, top_k=top_k,
        title=f"Injured Sample Saliency ({method.title()})",
        save_path=f"{save_dir}/injured_sample_saliency.png" if save_dir else None
    )
    
    # Plot saliency for healthy sample
    print("\nPlotting saliency for healthy sample...")
    healthy_top_idx, healthy_importance = plot_timeseries_saliency(
        healthy_saliency, channels, sample_idx=0, top_k=top_k,
        title=f"Healthy Sample Saliency ({method.title()})",
        save_path=f"{save_dir}/healthy_sample_saliency.png" if save_dir else None
    )
    
    # Print top features for each sample
    print(f"\nTop {min(5, top_k)} features for injured sample:")
    top_injured = np.argsort(injured_importance)[-5:][::-1]
    for i, idx in enumerate(top_injured):
        print(f"  {i+1}. {channels[idx]}: {injured_importance[idx]:.4f}")
    
    print(f"\nTop {min(5, top_k)} features for healthy sample:")
    top_healthy = np.argsort(healthy_importance)[-5:][::-1]
    for i, idx in enumerate(top_healthy):
        print(f"  {i+1}. {channels[idx]}: {healthy_importance[idx]:.4f}")
    
    return {
        'injured_idx': injured_idx,
        'healthy_idx': healthy_idx,
        'injured_prediction': pred_injured,
        'healthy_prediction': pred_healthy,
        'injured_saliency': injured_saliency,
        'healthy_saliency': healthy_saliency,
        'injured_top_features': injured_top_idx,
        'healthy_top_features': healthy_top_idx,
        'injured_importance': injured_importance,
        'healthy_importance': healthy_importance
    }

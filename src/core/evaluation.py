"""This module contains functions for evaluating the performance of the model."""

from typing import Tuple, List, Optional, Union, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.base import ClassifierMixin

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

def pick_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "macro"
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
                               ts_input,  # Can be np.ndarray or List[np.ndarray] for bilateral models
                               meta_input: Optional[np.ndarray] = None,
                               target_class: Optional[int] = None,
                               method: str = "vanilla") -> np.ndarray:
    """Compute saliency maps for timeseries data with optional metadata.
    
    Works with single-input, multi-input (timeseries + metadata), and bilateral models.
    
    Args:
        model: Trained Keras model
        ts_input: Timeseries input array of shape (batch_size, time_steps, features)
                 OR list of two arrays for bilateral models [left, right]
        meta_input: Optional metadata input array of shape (batch_size, meta_features)
        target_class: Class to compute gradients for (0 or 1), if None uses predicted class
        method: Saliency method ('vanilla', 'grad_x_input')
    
    Returns:
        saliency_map: Gradient-based saliency map for timeseries input
                     For bilateral models, returns list of saliency maps [left_saliency, right_saliency]
    """
    # Check if this is a bilateral model (list of two timeseries inputs)
    is_bilateral = isinstance(ts_input, list) and len(ts_input) == 2 and len(model.inputs) == 2
    
    # Determine if model has metadata
    has_metadata = meta_input is not None and len(model.inputs) > 1 and not is_bilateral
    
    if is_bilateral:
        if method == "vanilla":
            return _compute_vanilla_bilateral_saliency(model, ts_input, target_class)
        elif method == "grad_x_input":
            vanilla_grads = _compute_vanilla_bilateral_saliency(model, ts_input, target_class)
            return [grad * inp for grad, inp in zip(vanilla_grads, ts_input)]
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'vanilla', 'grad_x_input'")
    else:
        if method == "vanilla":
            return _compute_vanilla_timeseries_saliency(model, ts_input, meta_input, target_class, has_metadata)
        elif method == "grad_x_input":
            vanilla_grads = _compute_vanilla_timeseries_saliency(model, ts_input, meta_input, target_class, has_metadata)
            return vanilla_grads * ts_input
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'vanilla', 'grad_x_input'")


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


def _compute_vanilla_bilateral_saliency(model: tf.keras.Model,
                                       ts_inputs: list,
                                       target_class: Optional[int]) -> list:
    """Internal function for vanilla gradient computation for bilateral models."""
    left_input, right_input = ts_inputs
    
    # Convert to tensors
    left_tensor = tf.Variable(left_input.astype(np.float32), dtype=tf.float32)
    right_tensor = tf.Variable(right_input.astype(np.float32), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch([left_tensor, right_tensor])
        predictions = model([left_tensor, right_tensor])
        
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
    
    gradients = tape.gradient(target_scores, [left_tensor, right_tensor])
    return [grad.numpy() for grad in gradients]


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
                           X_ts,
                           y: np.ndarray,
                           channels: List[str],
                           X_meta: Optional[np.ndarray] = None,
                           injured_idx: Optional[int] = None,
                           healthy_idx: Optional[int] = None,
                           method: str = "vanilla",
                           top_k: int = 10,
                           save_dir: Optional[str] = None) -> dict:
    """Analyze saliency for specific injured and healthy samples.
    
    Args:
        model: Trained Keras model
        X_ts: Timeseries test data (single array or list of two arrays for bilateral models or dict for multimodal models)
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
    
    # Check if this is a bilateral model
    is_bilateral = isinstance(X_ts, list) and len(X_ts) == 2 and len(model.inputs) == 2

    is_multimodal = isinstance(X_ts, dict) and len(X_ts) == 3 and len(model.inputs) == 3
    
    # Get model predictions
    if is_bilateral:
        # For bilateral models
        X_left, X_right = X_ts
        pred_injured = model.predict([X_left[injured_idx:injured_idx+1], X_right[injured_idx:injured_idx+1]])[0, 0]
        pred_healthy = model.predict([X_left[healthy_idx:healthy_idx+1], X_right[healthy_idx:healthy_idx+1]])[0, 0]
    elif is_multimodal:
        # For multi-modal models
        inputs = {}
        for key in X_ts:
            inputs[key] = X_ts[key][injured_idx:injured_idx+1]
        
        pred_injured = model.predict(inputs)[0, 0]
        pred_healthy = model.predict(inputs)[0, 0]
    elif X_meta is not None:
        # For single timeseries + metadata models
        pred_injured = model.predict([X_ts[injured_idx:injured_idx+1], X_meta[injured_idx:injured_idx+1]])[0, 0]
        pred_healthy = model.predict([X_ts[healthy_idx:healthy_idx+1], X_meta[healthy_idx:healthy_idx+1]])[0, 0]
    else:
        # For single timeseries models
        pred_injured = model.predict(X_ts[injured_idx:injured_idx+1])[0, 0]
        pred_healthy = model.predict(X_ts[healthy_idx:healthy_idx+1])[0, 0]
    
    print(f"Model predictions:")
    print(f"  Injured sample (true=1): {pred_injured:.4f}")
    print(f"  Healthy sample (true=0): {pred_healthy:.4f}")
    
    # Compute saliency for both samples
    if is_bilateral:
        # For bilateral models
        X_left, X_right = X_ts
        injured_saliency = compute_timeseries_saliency(
            model, [X_left[injured_idx:injured_idx+1], X_right[injured_idx:injured_idx+1]], 
            target_class=None, method=method
        )
        
        healthy_saliency = compute_timeseries_saliency(
            model, [X_left[healthy_idx:healthy_idx+1], X_right[healthy_idx:healthy_idx+1]],
            target_class=None, method=method
        )
    else:
        # For single timeseries or timeseries + metadata models
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
    
    # For bilateral models, determine which saliency map to use based on channels
    if is_bilateral and isinstance(injured_saliency, list):
        # Check if channels are for left side (contain "L_") or right side (contain "R_")
        if any("L_" in ch for ch in channels):
            # Left side analysis
            injured_saliency_plot = injured_saliency[0]  # Left saliency
            healthy_saliency_plot = healthy_saliency[0]   # Left saliency
            side_name = "Left"
        elif any("R_" in ch for ch in channels):
            # Right side analysis
            injured_saliency_plot = injured_saliency[1]  # Right saliency
            healthy_saliency_plot = healthy_saliency[1]   # Right saliency
            side_name = "Right"
        else:
            # Default to left if unclear
            injured_saliency_plot = injured_saliency[0]
            healthy_saliency_plot = healthy_saliency[0]
            side_name = "Left"
    else:
        # Single model or already processed
        injured_saliency_plot = injured_saliency
        healthy_saliency_plot = healthy_saliency
        side_name = ""
    
    # Plot saliency for injured sample
    print(f"\nPlotting saliency for injured sample{' (' + side_name + ' side)' if side_name else ''}...")
    injured_top_idx, injured_importance = plot_timeseries_saliency(
        injured_saliency_plot, channels, sample_idx=0, top_k=top_k,
        title=f"Injured Sample Saliency ({method.title()}){' - ' + side_name + ' Side' if side_name else ''}",
        save_path=f"{save_dir}/injured_sample_saliency{'_' + side_name.lower() if side_name else ''}.png" if save_dir else None
    )
    
    # Plot saliency for healthy sample
    print(f"\nPlotting saliency for healthy sample{' (' + side_name + ' side)' if side_name else ''}...")
    healthy_top_idx, healthy_importance = plot_timeseries_saliency(
        healthy_saliency_plot, channels, sample_idx=0, top_k=top_k,
        title=f"Healthy Sample Saliency ({method.title()}){' - ' + side_name + ' Side' if side_name else ''}",
        save_path=f"{save_dir}/healthy_sample_saliency{'_' + side_name.lower() if side_name else ''}.png" if save_dir else None
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
    
    # Prepare return dictionary
    result = {
        'injured_idx': injured_idx,
        'healthy_idx': healthy_idx,
        'injured_prediction': pred_injured,
        'healthy_prediction': pred_healthy,
        'injured_saliency': injured_saliency_plot if is_bilateral and isinstance(injured_saliency, list) else injured_saliency,
        'healthy_saliency': healthy_saliency_plot if is_bilateral and isinstance(healthy_saliency, list) else healthy_saliency,
        'injured_top_features': injured_top_idx,
        'healthy_top_features': healthy_top_idx,
        'injured_importance': injured_importance,
        'healthy_importance': healthy_importance
    }
    
    # Add bilateral-specific information
    if is_bilateral:
        result['is_bilateral'] = True
        result['analyzed_side'] = side_name if isinstance(injured_saliency, list) else 'Both'
        if isinstance(injured_saliency, list):
            result['full_injured_saliency'] = injured_saliency  # Both left and right
            result['full_healthy_saliency'] = healthy_saliency   # Both left and right
    else:
        result['is_bilateral'] = False
    
    return result


class BasePredictor(ABC):
    """Base class for model predictors with threshold-based binary classification."""
    
    def __init__(self, model: tf.keras.Model):
        """Initialize predictor with a trained model.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
    
    @abstractmethod
    def predict_proba(self, *inputs) -> np.ndarray:
        """Predict probabilities for given inputs.

        Args:
            *inputs: Variable input arguments
        
        Returns:
            Array of prediction probabilities
        """
        pass
    
    def predict(self, threshold: float, *inputs) -> np.ndarray:
        """Predict binary labels using threshold.
        
        Args:
            threshold: Decision threshold
            *inputs: Variable input arguments
            
        Returns:
            Binary predictions (0 or 1)
        """
        y_pred_proba = self.predict_proba(*inputs)
        return (y_pred_proba > threshold).astype(int).flatten()


class BilateralPredictor(BasePredictor):
    """Predictor for bilateral models that take left and right inputs simultaneously."""
    
    def predict_proba(self, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
        """Predict probabilities using bilateral inputs.
        
        Args:
            X_left: Left side input data
            X_right: Right side input data
            
        Returns:
            Array of prediction probabilities
        """
        return self.model.predict([X_left, X_right])


class UnilateralPredictor(BasePredictor):
    """Predictor for unilateral models that predict each side separately then aggregate."""
    
    def __init__(self, model: tf.keras.Model, aggregation: str = "mean", verbose: bool = False):
        """Initialize unilateral predictor.
        
        Args:
            model: Trained Keras model that takes single-side input
            aggregation: How to aggregate left/right predictions ("mean" or "max")
            verbose: Whether to print verbose output
        """
        super().__init__(model)
        if aggregation not in ["mean", "max"]:
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.aggregation = aggregation
        self.verbose = verbose

    def predict_proba(self, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
        """Predict probabilities using unilateral model on both sides.
        
        Args:
            X_left: Left side input data
            X_right: Right side input data
            
        Returns:
            Array of aggregated prediction probabilities
        """
        # Predict each side separately
        y_pred_proba_left = self.model.predict(X_left)
        y_pred_proba_right = self.model.predict(X_right)
        
        # Aggregate predictions
        if self.aggregation == "max":
            if self.verbose:
                print("Aggregating predictions with max method")
            return np.maximum(y_pred_proba_left, y_pred_proba_right)
        else:  # mean
            if self.verbose:
                print("Aggregating predictions with mean method")
            return 0.5 * (y_pred_proba_left + y_pred_proba_right)


class MultiModalPredictor(BasePredictor):
    """Predictor for multi-modal models that take left and right inputs and metadata tabular data simultaneously."""
    
    def predict_proba(self, X_left: np.ndarray, X_right: np.ndarray, X_meta: np.ndarray) -> np.ndarray:
        """Predict probabilities using bilateral inputs.
        
        Args:
            X_left: Left side input data
            X_right: Right side input data
            X_meta: Metadata input data
        Returns:
            Array of prediction probabilities
        """
        inputs = {
            "left": X_left,
            "right": X_right,
            "metadata": X_meta
        }
        return self.model.predict(inputs)


class BilateralSingleInputPredictor(BasePredictor):
    """Predictor for bilateral models that take left and right inputs simultaneously."""
    
    def predict_proba(self, X_ts: np.ndarray) -> np.ndarray:
        """Predict probabilities using bilateral inputs.
        
        Args:
            X_ts: Input data
            
        Returns:
            Array of prediction probabilities
        """
        return self.model.predict([X_ts])

class SklearnAPIPredictor(BasePredictor):
    """Predictor for single input models."""
    
    def predict_proba(self, X_ts: np.ndarray) -> np.ndarray:
        """Predict probabilities using sklearn API.
        
        Args:
            X: Input data
            
        Returns:
            Array of prediction probabilities
        """
        return self.model.predict_proba(X_ts)[:, 1] # Keep only the probability of the positive class


def plot_confusion_matrix(y_true, y_pred, labels, name):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Labels to display
        name: Name of the model
    """
    plt.figure()
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        display_labels=labels,
        labels=[1, 0]
    )
    cm_disp.ax_.xaxis.set_label_position('top')
    cm_disp.ax_.xaxis.set_ticks_position('top')
    cm_disp.ax_.xaxis.set_label_text('Predicted')
    cm_disp.ax_.yaxis.set_label_text('Actual')
    cm_disp.ax_.set_title(f'Confusion Matrix - {name}')

def plot_roc_curve(y_true, y_pred_proba, name):
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        name: Name of the model
    """
    plt.figure()
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='AUC = 0.5')

    roc_disp = RocCurveDisplay.from_predictions(
        y_true, 
        y_pred_proba, 
        name=name,
        ax=ax,
    )
    roc_disp.ax_.set_title('ROC Curve')

def plot_precision_recall_curve(y_true, y_pred_proba, name, test_prevalence):
    """Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        name: Name of the model
        test_prevalence: Test prevalence
    """
    plt.figure()
    pr_disp = PrecisionRecallDisplay.from_predictions(
        y_true,
        y_pred_proba,
        name=name
    )
    pr_disp.ax_.set_title('Precision-Recall Curve')
    baseline_label = f'Baseline (prevalence = {test_prevalence:.2f})'
    pr_disp.ax_.axhline(y=test_prevalence, linestyle='--', color='gray', label=baseline_label)
    pr_disp.ax_.autoscale(tight=True)
    pr_disp.ax_.legend()


def model_test_summary(model: tf.keras.Model | ClassifierMixin,
                        inputs: Any,
                        y_true: np.ndarray,
                        threshold: float = 0.5,
                        predictor: BasePredictor = None,
                        model_name: str = None,
                        save_predictions: bool = False,
                    ) -> None:
    """Plot test results for a model.
    
    Args:
        model: Trained Keras model
        inputs: Test data
        y_true: Test labels
        threshold: Decision threshold
        predictor: implementation of BasePredictor
    """
    
    # Handle different types of inputs: list/tuple, dict, or single element
    if isinstance(inputs, dict):
        y_pred_proba = predictor.predict_proba(**inputs)
        y_pred = predictor.predict(threshold, **inputs)
    elif isinstance(inputs, (list, tuple)):
        y_pred_proba = predictor.predict_proba(*inputs)
        y_pred = predictor.predict(threshold, *inputs)
    else:
        y_pred_proba = predictor.predict_proba(inputs)
        y_pred = predictor.predict(threshold, inputs)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    test_avg_precision = average_precision_score(y_true, y_pred_proba)
    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_auc_roc = roc_auc_score(y_true, y_pred_proba)
    test_prevalence = np.mean(y_true == 1)

    if model_name is None:
        model_name = model.name if hasattr(model, "name") else model.__class__.__name__

    print("="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"=== Test Prevalence: {100 * test_prevalence:.2f} ===")
    print(f"Test AUC-PR : {test_avg_precision:.4f}")
    print(f"Test AUC-ROC : {test_auc_roc:.4f}")
    print("="*50)
    print(f"=== Threshold Dependent Metrics ===")
    print(f"=== Threshold: {threshold:.2f} ===")
    print("="*50)
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1-Macro Score : {test_f1:.4f}")
    print(f"Test Precision : {test_precision:.4f}")
    print(f"Test Recall : {test_recall:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, labels=[1,0], target_names=['Injured', 'Not Injured']))

    plot_confusion_matrix(y_true, y_pred, labels=['Injured', 'Not Injured'], name=model_name)
    plot_roc_curve(y_true, y_pred_proba, name=model_name)
    plot_precision_recall_curve(y_true, y_pred_proba, name=model_name, test_prevalence=test_prevalence)

    result = {
        "accuracy": test_accuracy,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc_roc": test_auc_roc,
        "auc_pr": test_avg_precision,
        "prevalence": test_prevalence,
        "threshold": threshold
    }

    if save_predictions:
        result["y_pred_proba"] = y_pred_proba
        result["y_pred"] = y_pred
        

    return result


def _take(a, idx):
    return a.iloc[idx] if hasattr(a, "iloc") else a[idx]


def stratified_group_train_val_test_split_ts(
    X,
    y,
    groups,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple:
    """Split X, y into train/val/test with subject grouping and stratification.

    - Test is taken first from full data using StratifiedGroupKFold.
    - Validation is taken from remaining train+val using StratifiedGroupKFold.
    - ``val_size`` is interpreted as a fraction of the original dataset; it is
      converted to a fraction of the remaining (1 - test_size).
    """
    assert 0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1

    # Outer split: train_val vs test
    n_splits_test = max(2, int(round(1.0 / test_size)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=shuffle, random_state=random_state)
    train_val_idx, test_idx = next(sgkf_test.split(X, y, groups))

    X_train_val, y_train_val, groups_train_val = _take(X, train_val_idx), _take(y, train_val_idx), _take(groups, train_val_idx)
    X_test, y_test, groups_test = _take(X, test_idx), _take(y, test_idx), _take(groups, test_idx)

    # Inner split on remaining: train vs val
    val_size_relative = val_size / (1.0 - test_size)
    n_splits_val = max(2, int(round(1.0 / val_size_relative)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=shuffle, random_state=random_state)
    train_idx, val_idx = next(sgkf_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, y_train, groups_train = _take(X_train_val, train_idx), _take(y_train_val, train_idx), _take(groups_train_val, train_idx)
    X_val, y_val, groups_val = _take(X_train_val, val_idx), _take(y_train_val, val_idx), _take(groups_train_val, val_idx)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
    )


def _standardise(X_ts, scaler_ts):
    """Standardise timeseries data."""
    N, T, C = X_ts.shape
    X_ts_reshaped = X_ts.reshape(-1, C)
    X_ts_scaled = scaler_ts.transform(X_ts_reshaped).reshape(N, T, C).astype(np.float32)
    return X_ts_scaled

def standardise_and_split_ts(X_ts, y, groups, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42, shuffle: bool = True) -> Tuple:
    """Standardise and split data into train/val/test with subject grouping and stratification.

    - Test is taken first from full data using StratifiedGroupKFold.
    - Validation is taken from remaining train+val using StratifiedGroupKFold.
    - ``val_size`` is interpreted as a fraction of the original dataset; it is
      converted to a fraction of the remaining (1 - test_size).
    
    Returns:
        Tuple:
            - X_ts_train_scaled: Standardised train data
            - X_ts_val_scaled: Standardised validation data
            - X_ts_test_scaled: Standardised test data
            - y_train: Standardised train labels
            - y_val: Standardised validation labels
            - y_test: Standardised test labels
            - subject_train: Standardised train subject IDs
            - subject_val: Standardised validation subject IDs
            - subject_test: Standardised test subject IDs
        - scaler_ts: Scaler object
    """
    (
    X_ts_train, X_ts_val, X_ts_test,
    y_train, y_val, y_test,
    subject_train, subject_val, subject_test,
    ) = stratified_group_train_val_test_split_ts(
        X_ts, y, groups, test_size=test_size, val_size=val_size, random_state=random_state, shuffle=shuffle
    )

    # Fit scaler on train data
    scaler_ts = StandardScaler()
    _, _, C = X_ts_train.shape
    scaler_ts.fit(X_ts_train.reshape(-1, C))

    # Transform all data
    X_ts_train_scaled = _standardise(X_ts_train, scaler_ts)
    X_ts_val_scaled = _standardise(X_ts_val, scaler_ts)
    X_ts_test_scaled = _standardise(X_ts_test, scaler_ts)

    return (
        X_ts_train_scaled, X_ts_val_scaled, X_ts_test_scaled,
        y_train, y_val, y_test,
        subject_train, subject_val, subject_test,
    ), scaler_ts


def verify_splits(X_ts,X_ts_train, X_ts_val, X_ts_test, y_train, y_val, y_test, subject_train, subject_val, subject_test):
    """Verify that the splits are correct."""
    print("\nVerifying splits...")
    print("="*50)

    # Check that there is no overlap in groups (subject_id) between splits
    train_groups_set = set(subject_train)
    val_groups_set = set(subject_val)
    test_groups_set = set(subject_test)

    overlap_train_val = train_groups_set & val_groups_set
    overlap_train_test = train_groups_set & test_groups_set
    overlap_val_test = val_groups_set & test_groups_set

    print(f"Overlap between train and val groups: {overlap_train_val}")
    print(f"Overlap between train and test groups: {overlap_train_test}")
    print(f"Overlap between val and test groups: {overlap_val_test}")

    assert len(overlap_train_val) == 0, "Train and Val groups overlap!"
    assert len(overlap_train_test) == 0, "Train and Test groups overlap!"
    assert len(overlap_val_test) == 0, "Val and Test groups overlap!"

    # Check that the total number of samples matches the original
    n_total = len(X_ts)
    n_split = len(X_ts_train) + len(X_ts_val) + len(X_ts_test)
    print(f"Total samples: {n_total}, Sum of splits: {n_split}")
    assert n_total == n_split, "Total number of samples does not match after split!"

    # Check class distribution in each split
    def print_class_distribution(y, name):
        unique, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        dist = {k: (v, 100.0 * v / total) for k, v in zip(unique, counts)}
        print(f"{name} class distribution:")
        for cls, (count, pct) in dist.items():
            print(f"  Class {cls}: {count} ({pct:.2f}%)")

    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Val")
    print_class_distribution(y_test, "Test")



def standardise_and_split(
        X: pd.DataFrame,
        Y: pd.Series,
        groups: Any,
        feature_categorical_columns: List[str] = None,
        num_fill_value: float = None,
        cat_drop: str = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
        return_pipeline: bool = False,
    ) -> Any:
    """
    Split X, Y into train/val/test with subject grouping and stratification and
    apply leak-free preprocessing (imputation, one-hot encoding, scaling).

    - Test is taken first from full data using StratifiedGroupKFold.
    - Validation is taken from remaining train+val using StratifiedGroupKFold.
    - Imputer and preprocessing (OHE + StandardScaler) are FIT on TRAIN ONLY.
    - The same fitted transformers are used to transform val and test.

    Args:
        X: Feature dataframe
        Y: Target series
        groups: Group labels (e.g., subject IDs) for group-aware splitting
        feature_categorical_columns: Optional explicit list of categorical columns
        num_fill_value: Optional constant for numeric imputation (defaults to median if None)
        cat_drop: OneHotEncoder drop policy (default: "if_binary")
        test_size: Fraction for test split
        val_size: Fraction for validation split (with respect to original dataset)
        random_state: RNG seed
        shuffle: Whether to shuffle before split
        return_pipeline: If True, also return the fitted (imputer, preprocessing)

    Returns:
        Either:
            (X_train_prep, X_val_prep, X_test_prep,
             y_train, y_val, y_test,
             groups_train, groups_val, groups_test)
        Or if return_pipeline:
            ((...same as above...), (imputer, preprocessing))
    """
    # Identify categorical and numerical columns
    if feature_categorical_columns is not None:
        cat_cols = feature_categorical_columns
    else:
        cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
    ) = stratified_group_train_val_test_split_ts(
        X, Y, groups,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # NOTE: Imputation and preprocessing are fitted on train only to avoid leakage.
    # Step 1: Imputation
    imputer = ColumnTransformer(
        transformers=[
            ('impute_num', SimpleImputer(strategy='median', copy=True, fill_value=num_fill_value), num_cols),
            ('impute_cat', SimpleImputer(strategy='constant', fill_value="unknown", copy=True), cat_cols),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    imputer.fit(X_train, y_train)

    def _impute_to_df(X_part: pd.DataFrame, index_like) -> pd.DataFrame:
        X_imp = imputer.transform(X_part)
        return pd.DataFrame(X_imp, columns=imputer.get_feature_names_out(), index=index_like)

    X_train_imp = _impute_to_df(X_train, X_train.index)
    X_val_imp   = _impute_to_df(X_val, X_val.index)
    X_test_imp  = _impute_to_df(X_test, X_test.index)

    # Step 2: Preprocessing (scale nums + one-hot cats).
    preprocessing = ColumnTransformer(
        transformers=[
            ('scale_num', StandardScaler(), num_cols),
            ('encode_cat', OneHotEncoder(drop=cat_drop, handle_unknown="ignore"), cat_cols),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    preprocessing.fit(X_train_imp, y_train)

    def _prep_to_df(X_imp_df: pd.DataFrame, index_like) -> pd.DataFrame:
        X_prep = preprocessing.transform(X_imp_df)
        X_df = pd.DataFrame(X_prep, columns=preprocessing.get_feature_names_out(), index=index_like)
        # Drop imputation helper columns if present
        unknown_cols_to_drop = [col + "_unknown" for col in cat_cols if col + "_unknown" in X_df.columns]
        return X_df.drop(columns=unknown_cols_to_drop) if unknown_cols_to_drop else X_df

    X_train_prep = _prep_to_df(X_train_imp, X_train.index)
    X_val_prep   = _prep_to_df(X_val_imp, X_val.index)
    X_test_prep  = _prep_to_df(X_test_imp, X_test.index)

    result = (
        X_train_prep, X_val_prep, X_test_prep,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
    )

    if return_pipeline:
        return result, (imputer, preprocessing)
    return result
"""Utilities for computing and visualizing feature importance across models."""

from math import ceil, sqrt
from typing import Dict, Tuple, Iterable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .processing import ResultsModel
from .explainability import get_feature_importance


def load_trained_models_and_results(results_summary_file: str) -> Tuple[Dict, Dict]:
    """Load results summary and corresponding trained models from disk.

    Args:
        results_summary_file: Path to the JSON results summary file.

    Returns:
        Tuple of two dictionaries:
            - results: Mapping model_name -> metrics and metadata
            - trained_models: Mapping model_name -> fitted sklearn Pipeline
    """
    with open(results_summary_file, "r") as f:
        results_json = f.read()

    results_model_loaded = ResultsModel.model_validate_json(results_json)

    results: Dict[str, Dict] = {}
    trained_models: Dict[str, object] = {}
    for result in results_model_loaded.results:
        results[result.model_name] = {
            "test_accuracy": result.test_accuracy,
            "test_precision": result.test_precision,
            "test_recall": result.test_recall,
            "test_f1": result.test_f1,
            "test_roc_auc": result.test_roc_auc,
            "y_pred": np.array(result.y_pred),
            "y_pred_proba": np.array(result.y_pred_proba),
            "best_parameters": result.best_parameters,
            "best_score": result.best_score,
        }
        trained_models[result.model_name] = joblib.load(result.model_file)

    return results, trained_models


def load_dataset(
    x_train_file: str,
    x_test_file: str,
    y_train_file: str,
    y_test_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train/test datasets from CSV files.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(x_train_file, index_col=0)
    X_test = pd.read_csv(x_test_file, index_col=0)
    y_train = pd.read_csv(y_train_file, index_col=0).squeeze()
    y_test = pd.read_csv(y_test_file, index_col=0).squeeze()
    return X_train, X_test, y_train, y_test


def compute_feature_importances_for_models(
    trained_models: Dict[str, object],
    X_data: pd.DataFrame,
    y_data: pd.Series,
    random_state: int = 42,
    use_permutation_importance: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Compute feature importances for multiple trained models.

    Args:
        trained_models: Mapping model_name -> fitted sklearn Pipeline
        X_data: Feature matrix to evaluate importances on (e.g., X_test)
        y_data: Target vector aligned with X_data
        random_state: Random state used by permutation importance when needed

    Returns:
        Mapping model_name -> DataFrame with columns [feature, importance, importance_type]
    """
    all_feature_importance: Dict[str, pd.DataFrame] = {}
    for model_name, model in trained_models.items():
        print(f"Extracting feature importance for {model_name}...")
        importance_df = get_feature_importance(
            model_name, model, X_data, y_data, random_state=random_state, use_permutation_importance=use_permutation_importance
        )
        all_feature_importance[model_name] = importance_df
    return all_feature_importance


def plot_feature_importances_by_model(
    all_feature_importance: Dict[str, pd.DataFrame],
    results: Dict[str, Dict],
    top_n: int = 15,
    figsize: Tuple[int, int] = (20, 16),
) -> None:
    """Plot horizontal bar charts of top-N feature importances per model.

    Colors bars by ROC-AUC thresholds for quick performance context.
    """
    n_models = len(all_feature_importance)
    if n_models == 0:
        print("No feature importances to plot.")
        return

    n_cols = ceil(sqrt(n_models))
    n_rows = ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    # Define color mapping and legend labels
    color_map = [
        ("darkgreen", "ROC-AUC > 0.8"),
        ("orange", "0.77 < ROC-AUC ≤ 0.8"),
        ("lightcoral", "ROC-AUC ≤ 0.77"),
    ]

    for idx, (model_name, importance_df) in enumerate(all_feature_importance.items()):
        ax = axes[idx]
        top_features = importance_df.head(top_n)
        bars = ax.barh(range(len(top_features)), top_features["importance"]) 
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"], fontsize=10)
        ax.set_xlabel("Importance Score")
        ax.set_title(
            f"{model_name}\n({top_features['importance_type'].iloc[0]} Importance)",
            fontsize=12,
        )
        ax.invert_yaxis()

        model_auc = results[model_name]["test_roc_auc"]
        if model_auc > 0.8:
            color = "darkgreen"
        elif model_auc > 0.77:
            color = "orange"
        else:
            color = "lightcoral"

        for bar in bars:
            bar.set_color(color)

        ax.grid(axis="x", alpha=0.3)

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a legend for the color coding
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=label) for c, label in color_map]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        title="Model ROC-AUC",
        fontsize=12,
        title_fontsize=13,
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.suptitle("Feature Importance Across All Models", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target_grid(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    features: Iterable[str],
    target_col: str = "is_injured",
    n_cols: int = 3,
    figsize_per_cell: Tuple[int, int] = (6, 5),
) -> None:
    """Plot categorical counts or boxplots of features against a binary target.

    For low-cardinality discrete features, plots countplots. For continuous
    features, plots boxplots by target class.
    """
    # Prepare combined DataFrame
    data = X.copy()
    if isinstance(y, pd.DataFrame):
        if target_col in y.columns:
            data[target_col] = y[target_col]
        elif y.shape[1] == 1:
            data[target_col] = y.iloc[:, 0]
        else:
            raise ValueError(
                f"Ambiguous target in DataFrame: specify a column named '{target_col}'."
            )
    else:
        data[target_col] = y

    features = list(features)
    n_features = len(features)
    n_rows = ceil(n_features / n_cols)
    figsize = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for i, feature in enumerate(features):
        ax = axes[i]
        series = data[feature]
        if series.nunique() <= 5 and series.dtype in [int, bool, object]:
            sns.countplot(data=data, x=feature, hue=target_col, ax=ax)
            ax.set_title(f"{feature} vs {target_col} (count)")
            ax.set_ylabel("Count")
        else:
            sns.boxplot(data=data, x=target_col, y=feature, ax=ax)
            ax.set_title(f"{feature} by {target_col} (boxplot)")
            ax.set_ylabel("Value")
        ax.set_xlabel(feature)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("Top Features vs Target", fontsize=18, y=1.03)
    plt.show()

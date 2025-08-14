"""Model selection and evaluation utilities"""
from typing import Dict, Tuple, Any, List, Optional, Generator

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

from .processing import ResultsModel, ResultModel


def split_train_test_set(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets.

    Args:
        X: Features.
        y: Labels.
        test_size: Size of the test set.
        random_state: Random state.
        stratify: Whether to stratify the data.
        groups: Optional groups to use when stratifying.

    Returns:
        Tuple of train and test sets.
    """
    if groups is not None:
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / test_size),  # Will get us roughly the desired test size.
            shuffle=True,
            random_state=random_state,
        )
        # Get the first split
        train_idx, test_idx = next(sgkf.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test, y_train, y_test
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )


def get_cv_splits(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        shuffle: bool,
        random_state: int = 42,
        groups: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Get CV splits.
    
    Args:
        X: Features.
        y: Labels.
        n_splits: Number of CV splits.
        shuffle: Whether to shuffle the data.
        random_state: Random state.
        groups: Optional groups to use when stratifying.
    
    Returns:
        Generator of CV splits, stratified by group, variable or not.
    """

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return cv.split(X, y)


def train_model_with_cv_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_config: Dict[str, Dict[str, Any]],
    groups: Optional[pd.Series] = None,
    scoring: str = "roc_auc",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Dict[str, GridSearchCV]:
    """Fit GridSearchCV pipelines using StratifiedGroupKFold splits.

    Args:
        X_train: Training features.
        y_train: Training labels.
        models_config: Mapping produced by ``get_baseline_models_config``.
        groups: Optional groups to use when stratifying.
        scoring: Metric for model selection.
        n_splits: Number of CV folds.
        shuffle: Whether to shuffle groups before split.
        random_state: Seed used when shuffling.
        n_jobs: Parallel jobs for GridSearchCV.
        verbose: Verbosity for GridSearchCV.

    Returns:
        Mapping from model name to fitted GridSearchCV.
    """
    cv_splits = get_cv_splits(
        X_train, y_train, n_splits, shuffle, random_state, groups=groups
    )
    cv_splits_list = list(cv_splits)  # Convert to list to ensure it is serializable for pickle...

    trained_models: Dict[str, GridSearchCV] = {}

    for model_name, config in models_config.items():
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv_splits_list,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        grid_search.fit(X_train, y_train)
        trained_models[model_name] = grid_search

    return trained_models


def evaluate_models(
    trained_models: Dict[str, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """Compute predictions and classification metrics for fitted models.

    Returns a mapping with keys: ``test_accuracy``, ``test_precision``,
    ``test_recall``, ``test_f1``, ``test_roc_auc``, ``y_pred``, ``y_pred_proba``,
    ``best_params``, and ``best_score``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, grid_search in trained_models.items():
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average="weighted")
        test_recall = recall_score(y_test, y_pred, average="weighted")
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        results[model_name] = {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "best_params": getattr(grid_search, "best_params_", None),
            "best_score": getattr(grid_search, "best_score_", None),
        }

    return results


def save_models_and_results(
    trained_models: Dict[str, GridSearchCV],
    eval_results: Dict[str, Dict[str, Any]],
    target_folder: str,
    results_summary_file: str,
) -> ResultsModel:
    """Persist fitted models and a summary JSON using pydantic models.

    Args:
        trained_models: Mapping of fitted GridSearchCV per model.
        eval_results: Mapping of computed metrics per model.
        target_folder: Directory where pickle files will be saved.
        results_summary_file: Path to the summary JSON file.

    Returns:
        The ``ResultsModel`` instance that was saved to disk.
    """
    os.makedirs(target_folder, exist_ok=True)

    model_filenames: Dict[str, str] = {}
    for model_name, model in trained_models.items():
        filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        model_path = os.path.join(target_folder, filename)
        joblib.dump(model, model_path)
        model_filenames[model_name] = model_path

    results_list: List[ResultModel] = []
    for model_name, result in eval_results.items():
        results_list.append(
            ResultModel(
                model_name=model_name,
                model_file=model_filenames[model_name],
                test_accuracy=float(result["test_accuracy"]),
                test_precision=float(result["test_precision"]),
                test_recall=float(result["test_recall"]),
                test_f1=float(result["test_f1"]),
                test_roc_auc=float(result["test_roc_auc"]),
                y_pred=result["y_pred"].tolist(),
                y_pred_proba=result["y_pred_proba"].tolist(),
                best_parameters=result.get("best_params"),
                best_score=float(result["best_score"]) if result.get("best_score") is not None else None,
            )
        )

    results_model = ResultsModel(results=results_list)

    os.makedirs(os.path.dirname(results_summary_file), exist_ok=True)
    with open(results_summary_file, "w") as f:
        f.write(results_model.model_dump_json(indent=2))

    return results_model


def load_models_and_results(results_summary_file: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load previously saved results JSON and corresponding model pickle files.

    Returns a tuple ``(results, trained_models)`` where ``results`` mirrors the
    structure returned by ``evaluate_models`` and ``trained_models`` maps model
    names to the loaded estimators.
    """
    with open(results_summary_file, "r") as f:
        results_json = f.read()

    results_model_loaded = ResultsModel.model_validate_json(results_json)

    results_loaded: Dict[str, Dict[str, Any]] = {}
    trained_models_loaded: Dict[str, Any] = {}

    for result in results_model_loaded.results:
        results_loaded[result.model_name] = {
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
        trained_models_loaded[result.model_name] = joblib.load(result.model_file)

    return results_loaded, trained_models_loaded


def plot_comparison_and_confusion(
    results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    class_labels: Optional[List[str]] = None,
    label_order: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Plot ROC curves and confusion matrices for model results.

    Args:
        results: Output of ``evaluate_models``.
        y_test: Ground truth labels.
        class_labels: Optional labels for confusion matrix display.
        label_order: Optional class order for confusion matrix.

    Returns:
        A dataframe comparing models on key metrics, sorted by ROC AUC.
    """
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="AUC = 0.5")

    comparison_data: List[Dict[str, Any]] = []
    for model_name, result in results.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Test ROC AUC": result["test_roc_auc"],
                "Test Accuracy": result["test_accuracy"],
                "Test Precision": result["test_precision"],
                "Test Recall": result["test_recall"],
                "Test F1-Score": result["test_f1"],
                "Test Best CV Score": result["best_score"],
            }
        )
        RocCurveDisplay.from_predictions(
            y_test,
            result["y_pred_proba"][:, 1],
            name=model_name,
            ax=ax,
        )
    ax.set_title("ROC Curve")

    for model_name, result in results.items():
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            result["y_pred"],
            display_labels=class_labels,
            labels=label_order,
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Test ROC AUC", ascending=False)
    return comparison_df

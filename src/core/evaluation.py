"""This module contains functions for evaluating the performance of the model."""

from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve


def pick_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, dict]:
    """Pick the threshold that maximizes the F1-Score.

    Args:
        y_true: (N,)
        y_score: (N,)

    Returns:
        threshold: float
        stats: dict
            best_f1: float
            P: float
            R: float
    """
    p, r, th = precision_recall_curve(y_true, y_score)  # th has len-1 compared to p/r
    f1 = 2*p*r/(p+r+1e-12)
    # align thresholds to same length
    th = np.append(th, th[-1])
    i = np.nanargmax(f1)
    return float(th[i]), {"best_f1": float(f1[i]), "P": float(p[i]), "R": float(r[i])}
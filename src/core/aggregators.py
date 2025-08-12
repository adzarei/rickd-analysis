from __future__ import annotations

from typing import Optional, List, Dict, Any

import dtw as dtw

import numpy as np
import kineticstoolkit as ktk

from .matlab_data_loader import MatlabDataLoader
from .timeseries import get_group_curves, apply_continuous_curve_registration


class BaseJointCurveAggregator:
    """
    Base aggregator for per-joint curves.

    Subclasses must implement _aggregate_group to transform a stacked group of
    curves with shape (n_cycles, 101, 3) into a single (101, 3) representation.
    """

    def aggregate_curves(self, curves: np.ndarray) -> np.ndarray:
        """Compute the representative curve for a set of curves by aggregating them.

        If multiple dimensions are provided, the curves are grouped by dimension.

        Args:
            curves: The Stack of curves to aggregate. (n_cycles, N, M)

        Returns:
            The representative curve as a (N, M) ndarray, where N is the number of points in the curve and M is the number of dimensions.
        """
        raise NotImplementedError


class MeanCurveAggregator(BaseJointCurveAggregator):
    """Aggregate by simple mean across cycles."""

    def aggregate_curves(self, curves: np.ndarray) -> np.ndarray:
        """Compute the representative curve for a set of curves by aggregating them.
        
        If multiple dimensions are provided, the curves are grouped by dimension.

        Aggregation Strategy: Mean.

        Args:
            curves: The Stack of curves to aggregate. (n_cycles, N, M)

        Returns:
            The representative curve as a (N, M) ndarray, where N is the number of points in the curve and M is the number of dimensions.
        """
        return curves.mean(axis=0)
    

class RegistrationCurveAggregator(BaseJointCurveAggregator):
    """Aggregate by per-axis continuous curve registration."""

    def __init__(self, **registration_kwargs: Dict[str, Any]) -> None:
        """Constructor for the registration curve aggregator.
        Arguments are passed as-is to the `apply_continuous_curve_registration` function.

        Args:
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.
            plot_dtw: Whether to plot the output alignment of the DTW algorithm.
            plot_rate: Plot every plot_rate iterations.
            plot_type: Type of plot to use.
            step_pattern: Step pattern to use for the alignment.
            loader: The loader to use. In case you need a local loader for parallel processing.
        """
        self.registration_kwargs = registration_kwargs or {}
    
    def aggregate_curves(self, curves: np.ndarray) -> np.ndarray:
        """Compute the representative curve for a set of curves by aggregating them.
        
        If multiple dimensions are provided, the curves are grouped by dimension.

        Aggregation Strategy: Curve registration.

        Args:
            curves: The Stack of curves to aggregate. (n_cycles, N, M)

        Returns:
            The representative curve as a (N, M) ndarray, where N is the number of points in the curve and M is the number of dimensions.
        """
        # curves: (n_cycles, N, M)
        # We want to apply registration per axis (M)
        registered_axes: List[np.ndarray] = []
        for axis_idx in range(curves.shape[2]):
            axis_curves = curves[:, :, axis_idx]  # (n_cycles, N)
            mean_axis = apply_continuous_curve_registration(curves=axis_curves, **self.registration_kwargs)
            registered_axes.append(mean_axis)
        return np.stack(registered_axes, axis=1)  # (N, M)

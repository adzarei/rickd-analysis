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

    def __init__(self, loader: Optional[MatlabDataLoader] = None) -> None:
        """Constuctor for the base aggregator.

        Args:
            loader: The loader to use. In case you need a local loader for parallel processing.
        """
        self.loader = loader

    def _verify_events(self, ts: ktk.TimeSeries, side: str, joint: str) -> None:
        """Verify that both right and left events exist for segmentation"""
        has_L_TD = any(e.name == 'L_TD' for e in ts.events)
        has_L_TO = any(e.name == 'L_TO' for e in ts.events)
        has_R_TD = any(e.name == 'R_TD' for e in ts.events)
        has_R_TO = any(e.name == 'R_TO' for e in ts.events)

        sid = ts.data_info['Metadata']['session_id'] or "<unknown>"
        if side == "L" and (not has_L_TD or not has_L_TO):
            raise ValueError(f"{side} {joint} for Session {sid} does not have L_TD or L_TO events")

        if side == "R" and (not has_R_TD or not has_R_TO):
            raise ValueError(f"{side} {joint} for Session {sid} does not have R_TD or R_TO events")
    
    def _verify_aggregation(self, group: np.ndarray) -> None:
        """Verify that the group of curves has the correct shape"""
        if group.shape != (101, 3):
            raise ValueError(f"Group of curves has incorrect shape: {group.shape}")

    def _load_joint_timeseries(self, session_id: str, joint: str, kind: str) -> tuple[ktk.TimeSeries, str]:
        if kind == "angle":
            ts = self.loader.get_joint_angles_timeseries(
                session_id, joint, normalized=False, include_events=True
            )
            data_key = f"{joint}_angle"
        elif kind == "velocity":
            ts = self.loader.get_joint_velocities_timeseries(
                session_id, joint, normalized=False, include_events=True
            )
            data_key = f"{joint}_velocity"
        else:
            raise ValueError("Invalid kind: expected 'angle' or 'velocity'")
        return ts, data_key

    def _aggregate_group(self, group: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_joint(self, session_id: str, joint: str, kind: str, loader: Optional[MatlabDataLoader] = None) -> np.ndarray:
        """Compute the representative stacked (angle XYZ + velocity XYZ per joint) matrix for a session.

        Args:
            session_id: The session id.
            joint: The joint name.
            kind: The kind of data to load.
            loader: The loader to use. In case you need a local loader for parallel processing.

        Returns:
            The representative stacked (angle XYZ + velocity XYZ per joint) matrix for a session.
        """
        # Initialize if i has not been provided in the constructor or in this function.
        if self.loader is None:
            if loader is None:
                loader = MatlabDataLoader()
            else:
                self.loader = loader
        
        ts, data_key = self._load_joint_timeseries(session_id, joint, kind)

        self._verify_events(ts, side=joint.split("_")[0], joint=joint)

        group = get_group_curves(ts, side=joint.split("_")[0], num_points=101, data_key=data_key)  # (n_cycles, 101, 3)
        group_representative = self._aggregate_group(group)

        self._verify_aggregation(group_representative)  # (101, 3)

        return group_representative


class MeanCurveAggregator(BaseJointCurveAggregator):
    """Aggregate by simple mean across cycles."""

    def _aggregate_group(self, group: np.ndarray) -> np.ndarray:
        return group.mean(axis=0)


class RegistrationCurveAggregator(BaseJointCurveAggregator):
    """Aggregate by per-axis continuous curve registration."""

    def __init__(self, loader: Optional[MatlabDataLoader] = None, **registration_kwargs: Dict[str, Any]) -> None:
        """Constructor for the registration curve aggregator.

        Args:
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.
            plot_dtw: Whether to plot the output alignment of the DTW algorithm.
            plot_rate: Plot every plot_rate iterations.
            plot_type: Type of plot to use.
            step_pattern: Step pattern to use for the alignment.
            loader: The loader to use. In case you need a local loader for parallel processing.
        """
        super().__init__(loader)
        self.registration_kwargs = registration_kwargs or {}

    def _aggregate_group(self, group: np.ndarray) -> np.ndarray:
        registered_axes: List[np.ndarray] = []
        for axis_idx in range(3):
            axis_curves = group[:, :, axis_idx]  # (n_cycles, 101)
            mean_axis = apply_continuous_curve_registration(axis_curves, **self.registration_kwargs)
            registered_axes.append(mean_axis)
        return np.stack(registered_axes, axis=1)  # (101, 3) 
from typing import Optional, Tuple, List

import kineticstoolkit as ktk

import numpy as np

from .matlab_data_loader import MatlabDataLoader
from .aggregators import (
    BaseJointCurveAggregator,
    MeanCurveAggregator,
)

from .timeseries import get_group_curves


def _load_joint_timeseries(loader: MatlabDataLoader, session_id: str, joint: str, kind: str) -> tuple[ktk.TimeSeries, str]:
    if kind == "angle":
        ts = loader.get_joint_angles_timeseries(
            session_id, joint, normalized=False, include_events=True
        )
        data_key = f"{joint}_angle"
    elif kind == "velocity":
        ts = loader.get_joint_velocities_timeseries(
            session_id, joint, normalized=False, include_events=True
        )
        data_key = f"{joint}_velocity"
    else:
        raise ValueError("Invalid kind: expected 'angle' or 'velocity'")
    return ts, data_key


def _verify_joint_ts(ts: ktk.TimeSeries, side: str, joint: str) -> None:
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


def _verify_aggregation(group: np.ndarray, num_points: int) -> None:
    """Verify that the group of curves has the correct shape"""
    if group.shape != (num_points, 3):
        raise ValueError(f"Group of curves has incorrect shape: {group.shape}")


def _process_joint(sid: str, joint: str, joint_side: str, kind: str, local_loader: MatlabDataLoader, aggregator: BaseJointCurveAggregator) -> np.ndarray:
    ts, data_key = _load_joint_timeseries(local_loader, sid, joint, kind)
    _verify_joint_ts(ts, side=joint_side, joint=joint)

    num_points = 101  # TODO: Make this configurable.
    group = get_group_curves(ts, side=joint_side, num_points=num_points, data_key=data_key)  # (n_cycles, num_points, 3)
    group_representative = aggregator.aggregate_curves(group)  # (num_points, 3)
    _verify_aggregation(group_representative, num_points)

    return group_representative


def compute_session_representation(
    sid: str,
    joints: List[str],
    aggregator: BaseJointCurveAggregator = MeanCurveAggregator()
) -> Optional[Tuple[str, np.ndarray]]:
    """
    Build the representative stacked (angle XYZ + velocity XYZ per joint) matrix for a session.

    Returns (session_id, (101, 6*len(joints)) array) or None on failure.
    """
    try:
        local_loader = MatlabDataLoader()

        per_joint: List[np.ndarray] = []
        for j in joints:
            for kind in ["angle", "velocity"]:

                # Pelvis does not have a side: pelvis_angle_X, pelvis_velocity_X, etc.
                joint_side = j.split("_")[0] if j.split("_")[0] in ["L", "R"] else ""

                if joint_side == "":
                    # Process first as R-side...
                    group_representative = _process_joint(sid, j, "R", kind, local_loader, aggregator)
                    per_joint.append(group_representative)
                    # ... then as L-side.
                    joint_side = "L"

                group_representative = _process_joint(sid, j, joint_side, kind, local_loader, aggregator)
                per_joint.append(group_representative)

        rep = np.concatenate(per_joint, axis=1)
        return sid, rep
    except Exception:
        return None

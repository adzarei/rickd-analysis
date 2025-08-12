from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

from .matlab_data_loader import MatlabDataLoader
from .aggregators import (
    BaseJointCurveAggregator,
    MeanCurveAggregator,
)


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
            ang = aggregator.compute_joint(sid, j, "angle", loader=local_loader)
            vel = aggregator.compute_joint(sid, j, "velocity", loader=local_loader)
            if ang.shape != (101, 3) or vel.shape != (101, 3):
                return None
            per_joint.append(ang)
            per_joint.append(vel)
        rep = np.concatenate(per_joint, axis=1)
        return sid, rep
    except Exception:
        return None

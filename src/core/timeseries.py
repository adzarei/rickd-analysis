"""This module contains helper functions for timeseries data."""

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import warnings
import numpy as np
import dtw as dtw
from typing import Optional

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

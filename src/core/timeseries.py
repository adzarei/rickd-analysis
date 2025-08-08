"""This module contains helper functions for timeseries data."""

import kineticstoolkit as ktk
import matplotlib.pyplot as plt


def get_group_curves(ts):
    """ Get the group of joint angle curves for a given session id and joint name."""
    NUM_POINTS = 101
    ts_normalized_on_stance = ktk.cycles.time_normalize(
        ts, event_name1="L_TD", event_name2="L_TO", n_points=NUM_POINTS
    )

    data = ktk.cycles.stack(ts_normalized_on_stance, n_points=NUM_POINTS)
    return data[next(iter(data))]  # Return first key


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

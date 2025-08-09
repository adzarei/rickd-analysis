"""This module contains functions to process marker data."""

import kineticstoolkit as ktk
from typing import Dict

LOWER_BODY_INTERCONNECTION = {
    "L_foot": {
        "Color": "b",
        "Links": [
            ["L_foot_1", "L_foot_2", "L_foot_3", "L_foot_1"],
            ["L_foot_1", "L_foot_4"],
            ["L_foot_2", "L_foot_4"],
            ["L_foot_3", "L_foot_4"],
            ["L_toe", "L_foot_3"],
            ["L_toe", "L_foot_2"],
            ["L_toe", "L_foot_1"],
        ],
    },
    "R_foot": {
        "Color": "r",
        "Links": [
            ["R_foot_1", "R_foot_2", "R_foot_3", "R_foot_1"],
            ["R_foot_1", "R_foot_4"],
            ["R_foot_2", "R_foot_4"],
            ["R_foot_3", "R_foot_4"],
            ["R_toe", "R_foot_3"],
            ["R_toe", "R_foot_2"],
            ["R_toe", "R_foot_1"],
        ],
    },
    "L_shank": {
        "Color": "g",
        "Links": [
            ["L_shank_1", "L_shank_2", "L_shank_3", "L_shank_4", "L_shank_1"],
        ],
    },
    "R_shank": {
        "Color": (1, 0.5, 0),
        "Links": [
            ["R_shank_1", "R_shank_2", "R_shank_3", "R_shank_4", "R_shank_1"],
        ],
    },
    "legs": {
        "Color": "w",
        "Links": [
            ["L_shank_3", "L_foot_1", "L_shank_4"],
            ["R_shank_3", "R_foot_1", "R_shank_4"],
        ],
    },
     "quads": {
        "Color": "w",
        "Links": [
            ["L_thigh_3", "L_shank_1"],
            ["L_thigh_4", "L_shank_2"],
            ["R_thigh_3", "R_shank_1"],
            ["R_thigh_4", "R_shank_2"],
            ["L_thigh_1", "pelvis_3"],
            ["L_thigh_2", "pelvis_1"],
            ["R_thigh_1", "pelvis_3"],
            ["R_thigh_2", "pelvis_2"],
        ],
    },
    "L_thigh": {
        "Color": (0.5, 0, 0.9),
        "Links": [
            ["L_thigh_1", "L_thigh_2", "L_thigh_3", "L_thigh_4", "L_thigh_1"],
        ],
    },
    "R_thigh": {
        "Color": (0, 1, 1),
        "Links": [
            ["R_thigh_1", "R_thigh_2", "R_thigh_3", "R_thigh_4", "R_thigh_1"],
        ],
    },
    "pelvis": {
        "Color": (1, 1, 0),
        "Links": [
            ["pelvis_1", "pelvis_2", "pelvis_3", "pelvis_1"],
            ["pelvis_4", "pelvis_1"],
            ["pelvis_4", "pelvis_2"],
            ["pelvis_4", "pelvis_3"],
        ],
    }
}


def visualize_marker_trajectories(ts: ktk.TimeSeries, playback_speed: float = 1, zoom: float = 0.6, up: str = "y", anterior: str = "-z", elevation: float = 0.2, azimuth: float = 0, font_size: float = 20, interconnections: Dict[str, Dict[str, str]] = None):
    """
    Visualize marker trajectories in a 3D plot.
    """
    if interconnections is None:
        interconnections = LOWER_BODY_INTERCONNECTION

    print("The player requires a interactive backend, please run the following command:")
    print("%matplotlib qt5")
    player = ktk.Player(ts, zoom=zoom, playback_speed=playback_speed, up=up, anterior=anterior, elevation=elevation, azimuth=azimuth, font_size=font_size)
    for group_key, interconnection in interconnections.items():
        player.interconnections[group_key] = interconnection
    player.play()

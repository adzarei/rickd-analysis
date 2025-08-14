"""
Comprehensive outlier detection module for time series gait analysis data.

This module provides multiple approaches for identifying and handling outliers
in irregular time series signals from biomechanical gait analysis.
"""

import numpy as np
import pandas as pd


def detect_temporal_anomalies(
    ts_data: np.ndarray,
    window_size: int = 10,
    threshold: float = 2.0,
    anomaly_threshold: float = 0.3
) -> np.ndarray:
    """
    Detect temporal anomalies in time series data using rolling z-scores.

    Args:
        ts_data: np.ndarray of shape (N, T, C), where
            N = number of samples,
            T = number of timepoints,
            C = number of channels.
        window_size: Size of the rolling window for local statistics.
        threshold: Z-score threshold for a timepoint to be considered anomalous.
        anomaly_threshold: Proportion of anomalous timepoints in a sample/channel
            required to flag the sample/channel as an outlier.

    Returns:
        outliers: Boolean np.ndarray of shape (N, C), True if sample/channel is an outlier.
    """
    N, T, C = ts_data.shape
    outliers = np.zeros((N, C), dtype=bool)
    for ch_idx in range(C):
        for sample_idx in range(N):
            timeseries = ts_data[sample_idx, :, ch_idx]
            df = pd.Series(timeseries)
            rolling_mean = df.rolling(window=window_size, center=True, min_periods=1).mean()
            rolling_std = df.rolling(window=window_size, center=True, min_periods=1).std()
            local_z_scores = np.abs((df - rolling_mean) / rolling_std)
            anomaly_ratio = np.mean(local_z_scores > threshold)
            outliers[sample_idx, ch_idx] = anomaly_ratio > anomaly_threshold
    return outliers


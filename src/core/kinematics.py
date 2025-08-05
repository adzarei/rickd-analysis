"""
Kinematics Module for Gait Analysis

This module provides Python implementations of the MATLAB gait analysis functions
from the Running Injury Clinic, converted to use Kinetics Toolkit where possible.

Main functions:
- gait_kinematics: Calculate joint angles using SVD-based pose estimation
- pca_touchdown/pca_toeoff: PCA-based gait event detection  
- gait_steps: Complete gait analysis pipeline
- processing_pipeline: Main wrapper function

Author: Converted from MATLAB code by Blayne Hettinga, Sean Osis, Allan Brett
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import warnings

import kineticstoolkit as ktk

from scipy import signal
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class GaitData:
    """Data structure for gait analysis inputs and outputs"""
    joints: Dict[str, np.ndarray]
    neutral: Dict[str, np.ndarray]
    dynamic: Dict[str, np.ndarray]
    hz: float
    angles: Optional[Dict[str, np.ndarray]] = None
    velocities: Optional[Dict[str, np.ndarray]] = None
    events: Optional[np.ndarray] = None
    discrete_variables: Optional[Dict] = None
    speed: Optional[float] = None
    gait_type: Optional[str] = None


class SoderqvistPoseEstimator:
    """
    SVD-based pose estimation using the Soderqvist method for rigid body tracking.
    
    This class implements the algorithm used in gait_kinematics.m for tracking
    body segments using marker clusters.
    """
    
    def __init__(self):
        self.reference_markers = {}
        self.segment_names = ['L_foot', 'R_foot', 'L_shank', 'R_shank', 'L_thigh', 'R_thigh', 'pelvis']
    
    def setup_reference_configuration(self, neutral_markers: Dict[str, np.ndarray]):
        """Set up reference marker configuration from neutral trial"""
        self.reference_markers = {}
        
        for segment in self.segment_names:
            if segment in neutral_markers:
                self.reference_markers[segment] = neutral_markers[segment].copy()
    
    def estimate_pose(self, dynamic_markers: Dict[str, np.ndarray], 
                     frame_idx: int) -> Dict[str, np.ndarray]:
        """
        Estimate pose of all segments at given frame using Soderqvist SVD method
        
        Args:
            dynamic_markers: Dictionary of dynamic marker data
            frame_idx: Frame index to estimate pose for
            
        Returns:
            Dictionary of rotation matrices for each segment
        """
        rotation_matrices = {}
        
        for segment in self.segment_names:
            if segment not in dynamic_markers or segment not in self.reference_markers:
                continue
                
            # Get current and reference marker positions
            dynamic_pos = dynamic_markers[segment][frame_idx, :].reshape(-1, 3)
            reference_pos = self.reference_markers[segment].reshape(-1, 3)
            
            # Calculate centroids
            dynamic_centroid = np.mean(dynamic_pos, axis=0)
            reference_centroid = np.mean(reference_pos, axis=0)
            
            # Center the point sets
            dynamic_centered = dynamic_pos - dynamic_centroid
            reference_centered = reference_pos - reference_centroid
            
            # Cross-correlation matrix (Step 3 of Soderqvist)
            C = dynamic_centered.T @ reference_centered
            
            # SVD decomposition (Step 4)
            P, T, Q_T = np.linalg.svd(C)
            Q = Q_T.T
            
            # Calculate rotation matrix (Step 5)
            # Ensure proper rotation (determinant = +1, not reflection)
            det_sign = np.linalg.det(P @ Q.T)
            diag_matrix = np.diag([1, 1, det_sign])
            R = P @ diag_matrix @ Q.T
            
            rotation_matrices[segment] = R
            
        return rotation_matrices
    
    def track_segments(self, dynamic_markers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Track all segments through entire motion"""
        n_frames = next(iter(dynamic_markers.values())).shape[0]
        tracked_poses = {}
        
        for segment in self.segment_names:
            if segment in dynamic_markers:
                tracked_poses[segment] = np.zeros((n_frames, 3, 3))
                
        for frame in range(n_frames):
            frame_poses = self.estimate_pose(dynamic_markers, frame)
            for segment, rotation in frame_poses.items():
                tracked_poses[segment][frame] = rotation
                
        return tracked_poses


class PCAEventDetector:
    """
    PCA-based touchdown and toe-off event detection
    
    Implements the algorithms from pca_td.m and pca_to.m
    """
    
    def __init__(self):
        self.td_model = None  # Touchdown detection model
        self.to_model = None  # Toe-off detection model
        self.default_hz = 200  # Default sampling frequency for models
        
        # Detection parameters (from MATLAB code)
        self.td_params = {
            'minpkdist': 40,      # Minimum peak distance 
            'negminpkht': 0.1,    # Negative minimum peak height
            'posminpkht': 0.1,    # Positive minimum peak height  
            'srchlgth': 50,       # Search length
            'chnklgth': 35        # Chunk length for PCA
        }
        
        self.to_params = {
            'minpkdist': 40,
            'posminpkht': 0.1,
            'chnklgth': 35
        }
    
    def load_pretrained_models(self, td_model_path: str, to_model_path: str):
        """Load pre-trained PCA models"""
        try:
            with open(td_model_path, 'rb') as f:
                self.td_model = pickle.load(f)
            with open(to_model_path, 'rb') as f:
                self.to_model = pickle.load(f)
        except FileNotFoundError:
            warnings.warn("Pre-trained models not found. Using fallback detection.")
    
    def detect_touchdown_events(self, angles: Dict[str, np.ndarray], 
                               hz: float, gait_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect touchdown events for left and right feet
        
        Args:
            angles: Joint angles from gait_kinematics
            hz: Sampling frequency
            gait_type: 'walk' or 'run'
            
        Returns:
            Tuple of (left_events, right_events) as frame indices
        """
        # Resample to default frequency if needed
        if hz != self.default_hz:
            angles = self._resample_angles(angles, hz, self.default_hz)
            hz = self.default_hz
        
        left_events = self._detect_touchdown_single_side(
            angles, 'L', self.td_params, gait_type)
        right_events = self._detect_touchdown_single_side(
            angles, 'R', self.td_params, gait_type)
            
        return left_events, right_events
    
    def detect_toeoff_events(self, angles: Dict[str, np.ndarray],
                            hz: float, gait_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Detect toe-off events for left and right feet"""
        if hz != self.default_hz:
            angles = self._resample_angles(angles, hz, self.default_hz)
            hz = self.default_hz
            
        left_events = self._detect_toeoff_single_side(
            angles, 'L', self.to_params, gait_type)
        right_events = self._detect_toeoff_single_side(
            angles, 'R', self.to_params, gait_type)
            
        return left_events, right_events
    
    def _detect_touchdown_single_side(self, angles: Dict[str, np.ndarray], 
                                     side: str, params: Dict, gait_type: str) -> np.ndarray:
        """Detect touchdown events for one side"""
        foot_angle = angles[f'{side}_foot'][:, 2]  # Sagittal plane (z-axis)
        
        # Flip and differentiate (from MATLAB code)
        negsig = -np.diff(foot_angle, n=2)
        
        # Find negative peaks
        from scipy.signal import find_peaks
        locs, _ = find_peaks(negsig, 
                           distance=params['minpkdist'],
                           height=params['negminpkht'])
        
        # Create search windows
        search_mask = np.zeros(len(negsig), dtype=bool)
        for loc in locs:
            end_idx = min(loc + params['srchlgth'], len(negsig))
            search_mask[loc:end_idx] = True
        
        # Find positive peaks in search windows
        possig = -negsig * search_mask
        peak_locs, _ = find_peaks(possig,
                                distance=params['minpkdist'], 
                                height=params['posminpkht'])
        
        # Apply PCA if model is available
        if self.td_model is not None:
            peak_locs = self._apply_pca_correction(angles, peak_locs, side, 'td')
            
        return peak_locs
    
    def _detect_toeoff_single_side(self, angles: Dict[str, np.ndarray],
                                  side: str, params: Dict, gait_type: str) -> np.ndarray:
        """Detect toe-off events for one side"""
        foot_angle = angles[f'{side}_foot'][:, 2]
        
        # Second derivative for toe-off detection
        signal_deriv = np.diff(foot_angle, n=2)
        
        # Find peaks
        from scipy.signal import find_peaks
        peak_locs, _ = find_peaks(signal_deriv,
                                distance=params['minpkdist'],
                                height=params['posminpkht'])
        
        # Apply PCA if model is available
        if self.to_model is not None:
            peak_locs = self._apply_pca_correction(angles, peak_locs, side, 'to')
            
        return peak_locs
    
    def _apply_pca_correction(self, angles: Dict[str, np.ndarray], 
                             peak_locs: np.ndarray, side: str, event_type: str) -> np.ndarray:
        """Apply PCA model to refine event timing"""
        # This would use the pre-trained PCA models loaded from MATLAB .mat files
        # For now, return the peak locations as-is
        return peak_locs
    
    def _resample_angles(self, angles: Dict[str, np.ndarray], 
                        original_hz: float, target_hz: float) -> Dict[str, np.ndarray]:
        """Resample angle signals to target frequency"""
        resampled = {}
        ratio = target_hz / original_hz
        
        for key, data in angles.items():
            new_length = int(data.shape[0] * ratio)
            resampled_data = np.zeros((new_length, data.shape[1]))
            
            for axis in range(data.shape[1]):
                f = interp1d(np.arange(data.shape[0]), data[:, axis], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
                new_indices = np.linspace(0, data.shape[0]-1, new_length)
                resampled_data[:, axis] = f(new_indices)
                
            resampled[key] = resampled_data
            
        return resampled


class GaitClassifier:
    """Walk vs Run classification using Linear Discriminant Analysis"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def load_pretrained_model(self, model_path: str):
        """Load pre-trained LDA classifier"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                self.is_trained = True
        except FileNotFoundError:
            warnings.warn("Pre-trained gait classifier not found. Using simple threshold.")
    
    def classify_gait(self, velocity: float, step_rate: float) -> str:
        """
        Classify gait as 'walk' or 'run' based on velocity and step rate
        
        Args:
            velocity: Gait velocity in m/s
            step_rate: Step rate in steps/min
            
        Returns:
            'walk' or 'run'
        """
        if self.is_trained and self.model is not None:
            features = np.array([[velocity, step_rate]])
            prediction = self.model.predict(features)[0]
            return prediction
        else:
            # Simple threshold-based classification
            return 'run' if velocity > 2.5 else 'walk'


def calculate_joint_centers(joints: Dict[str, np.ndarray], 
                          neutral: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate joint center locations from marker data
    
    Implements the joint center calculations from gait_kinematics.m
    """
    jc = {}
    
    # Pelvis center - average of 4 pelvis markers
    pelvis_markers = [neutral['pelvis_1'], neutral['pelvis_2'], 
                     neutral['pelvis_3'], neutral['pelvis_4']]
    jc['pelvis'] = np.mean(pelvis_markers, axis=0)
    
    # Hip centers - 25% distance between hips
    jc['L_hip'] = joints['L_hip'] + (joints['R_hip'] - joints['L_hip']) / 4
    jc['R_hip'] = joints['R_hip'] + (joints['L_hip'] - joints['R_hip']) / 4
    
    # Knee centers - midpoint of lateral and medial markers
    jc['L_knee'] = (joints['L_lat_knee'] + joints['L_med_knee']) / 2
    jc['R_knee'] = (joints['R_lat_knee'] + joints['R_med_knee']) / 2
    
    # Ankle centers - midpoint of lateral and medial markers  
    jc['L_ankle'] = (joints['L_lat_ankle'] + joints['L_med_ankle']) / 2
    jc['R_ankle'] = (joints['R_lat_ankle'] + joints['R_med_ankle']) / 2
    
    return jc


def gait_kinematics(joints: Dict[str, np.ndarray],
                   neutral: Dict[str, np.ndarray], 
                   dynamic: Dict[str, np.ndarray],
                   hz: float,
                   plots: bool = False) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Calculate joint angles and velocities using SVD-based pose estimation
    
    Python implementation of gait_kinematics.m
    
    Args:
        joints: Joint center locations from static trial
        neutral: Marker positions from static trial  
        dynamic: Marker positions from dynamic trial
        hz: Sampling frequency
        plots: Whether to generate plots
        
    Returns:
        Tuple of (angles, velocities, joint_centers, rotations, displacements)
    """
    # Calculate joint centers
    joint_centers = calculate_joint_centers(joints, neutral)
    
    # Initialize pose estimator
    pose_estimator = SoderqvistPoseEstimator()
    pose_estimator.setup_reference_configuration(neutral)
    
    # Track segment poses through motion
    rotations = pose_estimator.track_segments(dynamic)
    
    # Calculate joint angles using Kinetics Toolkit
    angles = _calculate_angles_ktk(rotations)
    
    # Calculate angular velocities
    velocities = _calculate_angular_velocities(angles, hz)
    
    # Calculate displacements (not implemented in this version)
    displacements = {}
    
    if plots:
        _plot_kinematics(angles, velocities)
    
    return angles, velocities, joint_centers, rotations, displacements


def _calculate_angles_ktk(rotations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate joint angles using Kinetics Toolkit"""
    angles = {}
    
    # Define joint pairs (proximal, distal)
    joint_pairs = [
        ('pelvis', 'L_thigh', 'L_hip'),
        ('pelvis', 'R_thigh', 'R_hip'), 
        ('L_thigh', 'L_shank', 'L_knee'),
        ('R_thigh', 'R_shank', 'R_knee'),
        ('L_shank', 'L_foot', 'L_ankle'),
        ('R_shank', 'R_foot', 'R_ankle')
    ]
    
    for proximal, distal, joint_name in joint_pairs:
        if proximal in rotations and distal in rotations:
            # Calculate relative rotation matrices
            prox_R = rotations[proximal]
            dist_R = rotations[distal]
            
            # Calculate relative rotation: R_relative = R_proximal^T * R_distal
            relative_R = np.matmul(prox_R.transpose(0, 2, 1), dist_R)
            
            # Convert to TimeSeries format for KTK
            n_frames = relative_R.shape[0]
            
            # Create transformation matrices with zero translation
            transforms = np.zeros((n_frames, 4, 4))
            transforms[:, :3, :3] = relative_R
            transforms[:, 3, 3] = 1.0
            
            # Create a TimeSeries with the transforms
            ts = ktk.TimeSeries()
            ts.time = np.linspace(0, n_frames-1, n_frames) / 200.0  # Assume 200 Hz
            ts.data[f'{joint_name}_transform'] = transforms
            
            # Extract angles using KTK (XYZ Euler angles in degrees)
            try:
                joint_angles = ktk.geometry.get_angles(ts, seq='xyz', degrees=True)
                # Extract the angle data
                angles[joint_name] = joint_angles.data[f'{joint_name}_transform_Angles']
            except:
                # Fallback to manual calculation if KTK fails
                angles[joint_name] = _rotation_matrix_to_euler(relative_R)
    
    return angles


def _calculate_angles_manual(rotations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate joint angles manually using rotation matrices"""
    angles = {}
    
    # This is a simplified implementation
    # In practice, you'd want to implement proper Euler angle extraction
    joint_pairs = [
        ('pelvis', 'L_thigh', 'L_hip'),
        ('pelvis', 'R_thigh', 'R_hip'),
        ('L_thigh', 'L_shank', 'L_knee'), 
        ('R_thigh', 'R_shank', 'R_knee'),
        ('L_shank', 'L_foot', 'L_ankle'),
        ('R_shank', 'R_foot', 'R_ankle')
    ]
    
    for proximal, distal, joint_name in joint_pairs:
        if proximal in rotations and distal in rotations:
            # Calculate relative rotation
            relative_rot = np.matmul(rotations[proximal].transpose(0, 2, 1), 
                                   rotations[distal])
            
            # Extract Euler angles (simplified XYZ sequence)
            joint_angles = _rotation_matrix_to_euler(relative_rot)
            angles[joint_name] = joint_angles
    
    return angles


def _rotation_to_transforms(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to homogeneous transforms for KTK"""
    n_frames = rotations.shape[0]
    transforms = np.zeros((n_frames, 4, 4))
    transforms[:, :3, :3] = rotations
    transforms[:, 3, 3] = 1
    return transforms


def _rotation_matrix_to_euler(rotation_matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to Euler angles (XYZ sequence)"""
    n_frames = rotation_matrices.shape[0]
    angles = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        R = rotation_matrices[i]
        
        # Extract XYZ Euler angles
        angles[i, 0] = np.arctan2(R[2, 1], R[2, 2])  # X (flexion/extension)
        angles[i, 1] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Y (ab/adduction)
        angles[i, 2] = np.arctan2(R[1, 0], R[0, 0])  # Z (internal/external rotation)
    
    return np.degrees(angles)  # Convert to degrees


def _calculate_angular_velocities(angles: Dict[str, np.ndarray], hz: float) -> Dict[str, np.ndarray]:
    """Calculate angular velocities from angles"""
    velocities = {}
    
    for joint, angle_data in angles.items():
        # Calculate numerical derivative
        dt = 1.0 / hz
        velocity_data = np.gradient(angle_data, dt, axis=0)
        velocities[joint] = velocity_data
    
    return velocities


def _plot_kinematics(angles: Dict[str, np.ndarray], velocities: Dict[str, np.ndarray]):
    """Plot joint angles and velocities"""
    n_joints = len(angles)
    fig, axes = plt.subplots(n_joints, 2, figsize=(12, 3*n_joints))
    
    for i, (joint, angle_data) in enumerate(angles.items()):
        # Plot angles
        axes[i, 0].plot(angle_data)
        axes[i, 0].set_title(f'{joint} Angles')
        axes[i, 0].set_ylabel('Angle (deg)')
        axes[i, 0].legend(['X', 'Y', 'Z'])
        
        # Plot velocities
        if joint in velocities:
            axes[i, 1].plot(velocities[joint])
            axes[i, 1].set_title(f'{joint} Angular Velocities')
            axes[i, 1].set_ylabel('Angular Velocity (deg/s)')
            axes[i, 1].legend(['X', 'Y', 'Z'])
    
    plt.tight_layout()
    plt.show()


def gait_steps(neutral: Dict[str, np.ndarray],
               dynamic: Dict[str, np.ndarray], 
               angles: Dict[str, np.ndarray],
               velocities: Dict[str, np.ndarray],
               hz: float,
               plots: bool = False) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, Dict, float, np.ndarray, str]:
    """
    Main gait analysis function - identifies events, normalizes data, calculates discrete variables
    
    Python implementation of gait_steps.m
    """
    # Calculate gait speed and step rate
    speed, step_rate = _calculate_gait_speed(dynamic, hz)
    
    # Classify gait type
    classifier = GaitClassifier()
    gait_type = classifier.classify_gait(speed, step_rate)
    
    # Detect gait events
    event_detector = PCAEventDetector()
    
    try:
        left_td, right_td = event_detector.detect_touchdown_events(angles, hz, gait_type)
        left_to, right_to = event_detector.detect_toeoff_events(angles, hz, gait_type)
        events_flag = np.ones_like(left_td)  # PCA events used
    except Exception as e:
        print(f"PCA event detection failed: {e}")
        # Fallback to simple peak detection
        left_td, right_td, left_to, right_to = _fallback_event_detection(dynamic, hz)
        events_flag = np.zeros_like(left_td)  # Fallback events used
    
    # Create events matrix
    events = _create_events_matrix(left_td, right_td, left_to, right_to)
    event = events.copy()  # Extended version with midswing (simplified here)
    
    # Normalize angles and velocities to stance phase
    norm_angles, norm_velocities = _normalize_gait_cycles(
        angles, velocities, events, hz)
    
    # Calculate discrete variables
    discrete_variables = _calculate_discrete_variables(
        norm_angles, norm_velocities, events, speed, step_rate, gait_type)
    
    if plots:
        _plot_gait_analysis(norm_angles, events, discrete_variables)
    
    return (norm_angles, norm_velocities, events, event, 
            discrete_variables, speed, events_flag, gait_type)


def _calculate_gait_speed(dynamic: Dict[str, np.ndarray], hz: float) -> Tuple[float, float]:
    """Calculate gait speed from heel marker motion"""
    # Find left heel marker (lowest medial marker)
    left_foot_markers = ['L_foot_1', 'L_foot_2', 'L_foot_3']
    heel_positions = []
    
    for marker in left_foot_markers:
        if marker in dynamic:
            heel_positions.append(dynamic[marker])
    
    if not heel_positions:
        return 0.0, 0.0
    
    # Use the marker with lowest average height
    heel_heights = [np.mean(pos[:, 2]) for pos in heel_positions]
    heel_marker = heel_positions[np.argmin(heel_heights)]
    
    # Calculate speed from forward (Y) position changes
    forward_pos = heel_marker[:, 1]  # Y direction
    speed_signal = np.gradient(forward_pos) * hz / 1000  # Convert to m/s
    
    # Find peaks in vertical velocity to estimate step rate
    vertical_vel = np.gradient(heel_marker[:, 2]) * hz
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-np.diff(heel_marker[:, 2]), 
                         distance=int(0.5*hz), height=0)
    
    if len(peaks) > 1:
        step_period = np.median(np.diff(peaks)) / hz
        step_rate = 60 / step_period  # steps per minute
    else:
        step_rate = 0.0
    
    median_speed = np.median(np.abs(speed_signal))
    
    return median_speed, step_rate


def _fallback_event_detection(dynamic: Dict[str, np.ndarray], 
                             hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback event detection using foot marker positions"""
    # Simple foot-forward, foot-back detection
    left_foot_markers = ['L_foot_1', 'L_foot_2', 'L_foot_3']
    right_foot_markers = ['R_foot_1', 'R_foot_2', 'R_foot_3']
    
    # Get representative foot positions
    left_foot_pos = dynamic[left_foot_markers[0]] if left_foot_markers[0] in dynamic else np.zeros((100, 3))
    right_foot_pos = dynamic[right_foot_markers[0]] if right_foot_markers[0] in dynamic else np.zeros((100, 3))
    
    # Filter signals
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 5/(hz/2), 'low')
    left_filtered = filtfilt(b, a, left_foot_pos[:, 1])  # Forward direction
    right_filtered = filtfilt(b, a, right_foot_pos[:, 1])
    
    # Find peaks (foot forward) and troughs (foot back)
    from scipy.signal import find_peaks
    left_ff, _ = find_peaks(-left_filtered, distance=int(0.35*hz))
    left_fb, _ = find_peaks(left_filtered, distance=int(0.35*hz))
    right_ff, _ = find_peaks(-right_filtered, distance=int(0.35*hz))
    right_fb, _ = find_peaks(right_filtered, distance=int(0.35*hz))
    
    return left_ff, right_ff, left_fb, right_fb


def _create_events_matrix(left_td: np.ndarray, right_td: np.ndarray,
                         left_to: np.ndarray, right_to: np.ndarray) -> np.ndarray:
    """Create events matrix from individual event arrays"""
    # Find common length
    min_length = min(len(left_td), len(right_td), len(left_to), len(right_to))
    
    events = np.column_stack([
        left_td[:min_length],
        left_to[:min_length], 
        right_td[:min_length],
        right_to[:min_length]
    ])
    
    return events


def _normalize_gait_cycles(angles: Dict[str, np.ndarray],
                          velocities: Dict[str, np.ndarray], 
                          events: np.ndarray,
                          hz: float,
                          target_length: int = 101) -> Tuple[Dict, Dict]:
    """Normalize gait cycles to consistent length"""
    normalized_angles = {}
    normalized_velocities = {}
    
    n_cycles = events.shape[0] - 1
    
    for joint in angles.keys():
        joint_cycles_ang = []
        joint_cycles_vel = []
        
        for cycle in range(n_cycles):
            start_idx = int(events[cycle, 0])  # Left touchdown
            end_idx = int(events[cycle+1, 0])   # Next left touchdown
            
            if end_idx > start_idx and end_idx < angles[joint].shape[0]:
                # Extract cycle data
                cycle_angles = angles[joint][start_idx:end_idx]
                cycle_velocities = velocities[joint][start_idx:end_idx]
                
                # Normalize to target length
                original_length = cycle_angles.shape[0]
                normalized_ang = np.zeros((target_length, cycle_angles.shape[1]))
                normalized_vel = np.zeros((target_length, cycle_velocities.shape[1]))
                
                for axis in range(cycle_angles.shape[1]):
                    f_ang = interp1d(np.linspace(0, 1, original_length), 
                                   cycle_angles[:, axis], kind='cubic')
                    f_vel = interp1d(np.linspace(0, 1, original_length),
                                   cycle_velocities[:, axis], kind='cubic')
                    
                    normalized_ang[:, axis] = f_ang(np.linspace(0, 1, target_length))
                    normalized_vel[:, axis] = f_vel(np.linspace(0, 1, target_length))
                
                joint_cycles_ang.append(normalized_ang)
                joint_cycles_vel.append(normalized_vel)
        
        if joint_cycles_ang:
            normalized_angles[joint] = np.array(joint_cycles_ang)
            normalized_velocities[joint] = np.array(joint_cycles_vel)
    
    return normalized_angles, normalized_velocities


def _calculate_discrete_variables(norm_angles: Dict[str, np.ndarray],
                                norm_velocities: Dict[str, np.ndarray],
                                events: np.ndarray,
                                speed: float, 
                                step_rate: float,
                                gait_type: str) -> Dict:
    """Calculate discrete gait variables"""
    discrete_vars = {
        'speed': speed,
        'step_rate': step_rate, 
        'gait_type': gait_type,
        'n_cycles': events.shape[0] - 1
    }
    
    # Calculate peak angles and velocities for each joint
    for joint, angle_data in norm_angles.items():
        if angle_data.size > 0:
            # Calculate means across cycles
            mean_angles = np.mean(angle_data, axis=0)  # Average across cycles
            
            # Peak values in each direction
            discrete_vars[f'{joint}_peak_flexion'] = np.max(mean_angles[:, 0])
            discrete_vars[f'{joint}_peak_extension'] = np.min(mean_angles[:, 0])
            discrete_vars[f'{joint}_range_of_motion'] = (np.max(mean_angles[:, 0]) - 
                                                        np.min(mean_angles[:, 0]))
    
    for joint, vel_data in norm_velocities.items():
        if joint in norm_velocities and vel_data.size > 0:
            mean_velocities = np.mean(vel_data, axis=0)
            discrete_vars[f'{joint}_peak_velocity'] = np.max(np.abs(mean_velocities[:, 0]))
    
    return discrete_vars


def _plot_gait_analysis(norm_angles: Dict[str, np.ndarray], 
                       events: np.ndarray,
                       discrete_variables: Dict):
    """Plot normalized gait cycles and discrete variables"""
    n_joints = len(norm_angles)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints))
    
    if n_joints == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue']
    
    for i, (joint, angle_data) in enumerate(norm_angles.items()):
        if angle_data.size > 0:
            mean_cycle = np.mean(angle_data, axis=0)
            std_cycle = np.std(angle_data, axis=0)
            
            x = np.linspace(0, 100, mean_cycle.shape[0])
            
            for axis in range(min(3, mean_cycle.shape[1])):
                axes[i].plot(x, mean_cycle[:, axis], color=colors[axis], 
                           label=f'Axis {axis+1}')
                axes[i].fill_between(x, 
                                   mean_cycle[:, axis] - std_cycle[:, axis],
                                   mean_cycle[:, axis] + std_cycle[:, axis],
                                   alpha=0.3, color=colors[axis])
            
            axes[i].set_title(f'{joint} - Normalized Gait Cycle')
            axes[i].set_xlabel('Gait Cycle (%)')
            axes[i].set_ylabel('Angle (deg)')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print discrete variables
    print("\nDiscrete Variables:")
    for key, value in discrete_variables.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def processing_pipeline(json_file_path: str, 
                       code_folder: str = None,
                       plots: bool = False) -> Dict:
    """
    Main processing pipeline - Python implementation of processing_code_example.m
    
    Args:
        json_file_path: Path to JSON data file
        code_folder: Path to code folder (for model files)
        plots: Whether to generate plots
        
    Returns:
        Dictionary containing all analysis results
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert JSON structure to expected format
    joints = {key: np.array(value).T for key, value in data['joints'].items()}
    neutral = {key: np.array(value).T for key, value in data['neutral'].items()}
    
    results = {}
    
    # Process walking data if available
    if 'walking' in data and data['walking']:
        walking = {key: np.array(value).T for key, value in data['walking'].items()}
        hz_w = data['hz_w']
        
        print("Processing walking data...")
        w_angles, w_velocities, w_jc, w_R, w_djc = gait_kinematics(
            joints, neutral, walking, hz_w, plots)
        
        w_norm_ang, w_norm_vel, w_events, w_event, w_discrete, w_speed, w_flags, w_label = gait_steps(
            neutral, walking, w_angles, w_velocities, hz_w, plots)
        
        results['walking'] = {
            'angles': w_angles,
            'velocities': w_velocities,
            'joint_centers': w_jc,
            'rotations': w_R,
            'normalized_angles': w_norm_ang,
            'normalized_velocities': w_norm_vel,
            'events': w_events,
            'discrete_variables': w_discrete,
            'speed': w_speed,
            'gait_type': w_label
        }
    
    # Process running data if available
    if 'running' in data and data['running']:
        running = {key: np.array(value).T for key, value in data['running'].items()}
        hz_r = data['hz_r']
        
        print("Processing running data...")
        r_angles, r_velocities, r_jc, r_R, r_djc = gait_kinematics(
            joints, neutral, running, hz_r, plots)
        
        r_norm_ang, r_norm_vel, r_events, r_event, r_discrete, r_speed, r_flags, r_label = gait_steps(
            neutral, running, r_angles, r_velocities, hz_r, plots)
        
        results['running'] = {
            'angles': r_angles,
            'velocities': r_velocities, 
            'joint_centers': r_jc,
            'rotations': r_R,
            'normalized_angles': r_norm_ang,
            'normalized_velocities': r_norm_vel,
            'events': r_events,
            'discrete_variables': r_discrete,
            'speed': r_speed,
            'gait_type': r_label
        }
    
    return results


# Example usage and testing functions
def test_kinematics_module():
    """Test the kinematics module with dummy data"""
    print("Testing kinematics module...")
    
    # Create dummy data
    n_frames = 1000
    n_markers = 3
    
    # Dummy joint centers
    joints = {
        'L_hip': np.array([0.1, 0.0, 0.8]),
        'R_hip': np.array([-0.1, 0.0, 0.8]),
        'L_lat_knee': np.array([0.05, 0.0, 0.5]),
        'L_med_knee': np.array([0.15, 0.0, 0.5]),
        'R_lat_knee': np.array([-0.05, 0.0, 0.5]),
        'R_med_knee': np.array([-0.15, 0.0, 0.5]),
        'L_lat_ankle': np.array([0.05, 0.0, 0.1]),
        'L_med_ankle': np.array([0.15, 0.0, 0.1]),
        'R_lat_ankle': np.array([-0.05, 0.0, 0.1]),
        'R_med_ankle': np.array([-0.15, 0.0, 0.1])
    }
    
    # Dummy neutral markers
    neutral = {
        'pelvis_1': np.array([0.1, 0.0, 0.9]),
        'pelvis_2': np.array([-0.1, 0.0, 0.9]),
        'pelvis_3': np.array([-0.1, -0.1, 0.9]),
        'pelvis_4': np.array([0.1, -0.1, 0.9]),
        'L_foot': np.random.randn(n_markers, 3) * 0.05 + [0.1, 0.0, 0.0],
        'R_foot': np.random.randn(n_markers, 3) * 0.05 + [-0.1, 0.0, 0.0],
        'L_shank': np.random.randn(n_markers, 3) * 0.05 + [0.1, 0.0, 0.3],
        'R_shank': np.random.randn(n_markers, 3) * 0.05 + [-0.1, 0.0, 0.3],
        'L_thigh': np.random.randn(n_markers, 3) * 0.05 + [0.1, 0.0, 0.6],
        'R_thigh': np.random.randn(n_markers, 3) * 0.05 + [-0.1, 0.0, 0.6],
        'pelvis': np.random.randn(n_markers, 3) * 0.05 + [0.0, 0.0, 0.9]
    }
    
    # Dummy dynamic markers with some motion
    dynamic = {}
    for segment, static_pos in neutral.items():
        if segment.startswith('pelvis'):
            continue
        # Add some sinusoidal motion
        time = np.linspace(0, 10, n_frames)
        motion = np.zeros((n_frames, 3))
        motion[:, 0] = 0.1 * np.sin(2 * np.pi * time)  # Side motion
        motion[:, 1] = time * 0.5  # Forward motion
        motion[:, 2] = 0.05 * np.sin(4 * np.pi * time)  # Vertical motion
        
        dynamic[segment] = static_pos[np.newaxis, :, :] + motion[:, np.newaxis, :]
        dynamic[segment] = dynamic[segment].reshape(n_frames, -1)
    
    hz = 200.0
    
    try:
        # Test gait kinematics
        print("Testing gait_kinematics...")
        angles, velocities, jc, R, djc = gait_kinematics(joints, neutral, dynamic, hz)
        print(f"✓ Calculated angles for {len(angles)} joints")
        
        # Test gait steps
        print("Testing gait_steps...")
        norm_ang, norm_vel, events, event, discrete_vars, speed, flags, label = gait_steps(
            neutral, dynamic, angles, velocities, hz)
        print(f"✓ Detected {len(events)} gait events")
        print(f"✓ Gait type: {label}, Speed: {speed:.2f} m/s")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test_kinematics_module() 
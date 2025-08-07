"""
Ground Reaction Force (GRF) estimation using biomechanical modeling.

This module provides functionality to estimate ground reaction forces from kinematic data
using both biorbd (if available) and simplified kinematic-based methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

try:
    import biorbd
    BIORBD_AVAILABLE = True
except ImportError:
    BIORBD_AVAILABLE = False
    warnings.warn("biorbd not available. Install with: conda install -c conda-forge biorbd")

from .matlab_data_loader import MatlabDataLoader
from .constants import RICKD_MATLAB_OUTPUT_FOLDER


class GRFEstimator:
    """
    Ground Reaction Force estimator using biomechanical modeling.
    
    This class provides methods to:
    1. Load kinematic data from MATLAB processing
    2. Set up biomechanical models (if biorbd available)
    3. Estimate GRF using various methods
    4. Analyze contact forces and moments
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 matlab_data_loader: Optional[MatlabDataLoader] = None):
        """
        Initialize the GRF estimator.
        
        Args:
            model_path: Path to the biorbd model file (.bioMod) - optional if biorbd not available
            matlab_data_loader: Optional MatlabDataLoader instance
        """
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        
        if BIORBD_AVAILABLE and model_path:
            if not self.model_path.exists():
                warnings.warn(f"Model file not found: {model_path}. Will use simplified methods only.")
            else:
                try:
                    # Load the biomechanical model
                    self.model = biorbd.Model(str(self.model_path))
                except Exception as e:
                    warnings.warn(f"Could not load biorbd model: {e}. Will use simplified methods only.")
        
        # Initialize data loader
        if matlab_data_loader is None:
            self.data_loader = MatlabDataLoader()
        else:
            self.data_loader = matlab_data_loader
            
        # Cache for processed data
        if self.model:
            self._joint_mapping = self._create_joint_mapping()
            self._marker_mapping = self._create_marker_mapping()
        else:
            self._joint_mapping = {}
            self._marker_mapping = {}
        
    def _create_joint_mapping(self) -> Dict[str, int]:
        """Create mapping between joint names and model indices."""
        if not self.model:
            return {}
        joint_mapping = {}
        for i in range(self.model.nbDof()):
            joint_name = self.model.nameDof()[i].to_string()
            joint_mapping[joint_name] = i
        return joint_mapping
        
    def _create_marker_mapping(self) -> Dict[str, int]:
        """Create mapping between marker names and model indices."""
        if not self.model:
            return {}
        marker_mapping = {}
        for i in range(self.model.nbMarkers()):
            marker_name = self.model.markerNames()[i].to_string()
            marker_mapping[marker_name] = i
        return marker_mapping
    
    def get_joint_kinematics(self, 
                           session_id: str, 
                           joints: Optional[List[str]] = None,
                           normalized: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load joint kinematics data for the specified session.
        
        Args:
            session_id: Session identifier
            joints: List of joint names to load (None for all available)
            normalized: Whether to load normalized data
            
        Returns:
            Dictionary mapping joint names to DataFrames with angles and velocities
        """
        if joints is None:
            joints = self.data_loader.get_available_joints(session_id)
            
        kinematics = {}
        for joint in joints:
            try:
                angles = self.data_loader.get_joint_angles(session_id, joint, normalized)
                velocities = self.data_loader.get_joint_velocities(session_id, joint, normalized)
                
                # Combine angles and velocities
                combined_df = angles.copy()
                for col in velocities.columns:
                    if col not in ['TimeIndex', 'PercentGaitCycle']:
                        # Rename velocity columns to distinguish from angles
                        vel_col_name = col.replace('_deg', '_vel_deg').replace('_per_s', '_per_s')
                        combined_df[vel_col_name] = velocities[col]
                        
                kinematics[joint] = combined_df
                
            except FileNotFoundError:
                warnings.warn(f"Joint data not found for {joint} in session {session_id}")
                continue
                
        return kinematics
    
    def convert_to_biorbd_format(self, 
                                kinematics: Dict[str, pd.DataFrame],
                                time_column: str = 'TimeIndex') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert kinematic data to biorbd format (Q, Qdot, Qddot).
        
        Args:
            kinematics: Dictionary of joint kinematics data
            time_column: Name of the time column
            
        Returns:
            Tuple of (Q, Qdot, Qddot) arrays where:
            - Q: joint positions [n_dof, n_frames]
            - Qdot: joint velocities [n_dof, n_frames] 
            - Qddot: joint accelerations [n_dof, n_frames]
        """
        # Get time series length from first joint
        first_joint_data = next(iter(kinematics.values()))
        n_frames = len(first_joint_data)
        n_dof = self.model.nbDof()
        
        # Initialize arrays
        Q = np.zeros((n_dof, n_frames))
        Qdot = np.zeros((n_dof, n_frames))
        Qddot = np.zeros((n_dof, n_frames))
        
        # Map joint data to model DOF
        joint_order = ['pelvis', 'L_hip', 'L_knee', 'L_ankle', 'R_hip', 'R_knee', 'R_ankle']
        dof_index = 0
        
        for joint_name in joint_order:
            if joint_name in kinematics:
                data = kinematics[joint_name]
                
                # Extract angles (assuming X, Y, Z columns)
                angle_cols = [col for col in data.columns if '_deg' in col and '_vel_' not in col]
                vel_cols = [col for col in data.columns if '_vel_deg' in col]
                
                for i, (angle_col, vel_col) in enumerate(zip(angle_cols, vel_cols)):
                    if dof_index < n_dof:
                        # Convert degrees to radians
                        Q[dof_index, :] = np.deg2rad(data[angle_col].values)
                        Qdot[dof_index, :] = np.deg2rad(data[vel_col].values)
                        
                        # Calculate accelerations using finite differences
                        if len(data) > 1:
                            dt = 1.0 / 100.0  # Assuming 100 Hz sampling rate
                            Qddot[dof_index, 1:] = np.diff(Qdot[dof_index, :]) / dt
                            Qddot[dof_index, 0] = Qddot[dof_index, 1]  # Extrapolate first frame
                            
                        dof_index += 1
                        
        return Q, Qdot, Qddot
    
    def estimate_contact_forces(self,
                              Q: np.ndarray,
                              Qdot: np.ndarray, 
                              Qddot: np.ndarray,
                              contact_threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Estimate contact forces using inverse dynamics.
        
        Args:
            Q: Joint positions [n_dof, n_frames]
            Qdot: Joint velocities [n_dof, n_frames]
            Qddot: Joint accelerations [n_dof, n_frames]
            contact_threshold: Threshold for contact detection (m)
            
        Returns:
            Dictionary with estimated contact forces for each contact point
        """
        n_frames = Q.shape[1]
        contact_forces = {}
        
        # Get contact point information
        n_contacts = self.model.nbContacts()
        contact_names = []
        for i in range(n_contacts):
            contact_names.append(f"contact_{i}")
            
        # Initialize contact force arrays
        for contact_name in contact_names:
            contact_forces[contact_name] = {
                'force': np.zeros((3, n_frames)),  # Fx, Fy, Fz
                'moment': np.zeros((3, n_frames)), # Mx, My, Mz
                'in_contact': np.zeros(n_frames, dtype=bool)
            }
        
        # Process each frame
        for frame in range(n_frames):
            q = Q[:, frame]
            qdot = Qdot[:, frame]
            qddot = Qddot[:, frame]
            
            # Update model state
            self.model.UpdateKinematicsCustom(q, qdot, qddot)
            
            # Detect contacts based on foot position
            contact_states = self._detect_contacts(q, contact_threshold)
            
            # Calculate inverse dynamics
            if np.any(contact_states):
                # Simple contact force estimation
                # This is a simplified approach - in practice, you'd use more sophisticated methods
                tau = self._calculate_joint_torques(q, qdot, qddot)
                forces = self._estimate_contact_forces_from_torques(q, tau, contact_states)
                
                # Distribute forces among active contacts
                active_contacts = np.where(contact_states)[0]
                for i, contact_idx in enumerate(active_contacts):
                    contact_name = contact_names[contact_idx]
                    # Simplified force distribution
                    contact_forces[contact_name]['force'][:, frame] = forces / len(active_contacts)
                    contact_forces[contact_name]['in_contact'][frame] = True
                    
        return contact_forces
    
    def _detect_contacts(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect ground contacts based on foot height.
        
        Args:
            q: Joint positions for current frame
            threshold: Contact threshold (m)
            
        Returns:
            Boolean array indicating contact state for each contact point
        """
        # Simple contact detection based on foot height
        # This assumes feet are the last segments and uses Y coordinate for height
        
        # Get foot positions
        n_contacts = self.model.nbContacts()
        contact_states = np.zeros(n_contacts, dtype=bool)
        
        # For simplicity, assume left and right foot contacts
        # You would need to implement proper contact detection based on your model
        if n_contacts >= 2:
            # Simplified: assume contacts when Y position is below threshold
            # This would need to be adapted based on your specific model structure
            contact_states[0] = True  # Left foot contact (placeholder)
            contact_states[1] = True  # Right foot contact (placeholder)
            
        return contact_states
    
    def _calculate_joint_torques(self, 
                               q: np.ndarray, 
                               qdot: np.ndarray, 
                               qddot: np.ndarray) -> np.ndarray:
        """
        Calculate joint torques using inverse dynamics.
        
        Args:
            q: Joint positions
            qdot: Joint velocities  
            qddot: Joint accelerations
            
        Returns:
            Joint torques array
        """
        # This is a simplified implementation
        # In practice, you'd use biorbd's inverse dynamics functions
        tau = np.zeros(self.model.nbDof())
        
        # Placeholder for inverse dynamics calculation
        # You would use biorbd's InverseDynamics function here
        # tau = self.model.InverseDynamics(q, qdot, qddot)
        
        return tau
    
    def _estimate_contact_forces_from_torques(self,
                                            q: np.ndarray,
                                            tau: np.ndarray, 
                                            contact_states: np.ndarray) -> np.ndarray:
        """
        Estimate contact forces from joint torques.
        
        Args:
            q: Joint positions
            tau: Joint torques
            contact_states: Contact states
            
        Returns:
            Estimated contact forces [Fx, Fy, Fz]
        """
        # Simplified contact force estimation
        # This would typically involve solving the contact constraint equations
        
        # Placeholder implementation
        forces = np.array([0.0, 800.0, 0.0])  # Typical vertical GRF during running
        
        return forces
    
    def estimate_grf_simplified(self, 
                              joint_angles: pd.DataFrame, 
                              joint_velocities: pd.DataFrame,
                              subject_mass: float = 70.0) -> Dict[str, np.ndarray]:
        """
        Simplified GRF estimation from joint kinematics.
        
        This method uses empirical relationships between joint motion and GRF
        commonly used in running biomechanics research.
        
        Args:
            joint_angles: DataFrame with joint angles
            joint_velocities: DataFrame with joint velocities  
            subject_mass: Subject mass in kg
            
        Returns:
            Dictionary with estimated GRF components
        """
        
        # Extract key joint data
        time_frames = len(joint_angles)
        dt = 1.0 / 100.0  # Assuming 100 Hz sampling
        
        # Initialize GRF arrays
        grf_vertical = np.zeros(time_frames)
        grf_anterior = np.zeros(time_frames)
        grf_lateral = np.zeros(time_frames)
        
        # Simple contact detection based on ankle angle pattern
        # During stance phase, ankle typically goes through dorsiflexion-plantarflexion
        ankle_angle = joint_angles['Z_deg'].values  # Sagittal plane (dorsi/plantar flexion)
        
        # Detect stance phase (simplified approach)
        # In real running, you'd use more sophisticated methods
        stance_threshold = np.mean(ankle_angle) - 0.5 * np.std(ankle_angle)
        in_stance = ankle_angle < stance_threshold
        
        # Estimate vertical GRF during stance
        if np.any(in_stance):
            stance_frames = np.where(in_stance)[0]
            stance_duration = len(stance_frames) * dt
            
            # Create typical running GRF pattern (double-peaked)
            for i, frame in enumerate(stance_frames):
                # Normalized stance time (0 to 1)
                t_norm = i / len(stance_frames)
                
                # Double-peaked vertical GRF pattern
                # Peak 1: Impact peak (~20% stance)
                # Peak 2: Propulsive peak (~80% stance)
                if t_norm < 0.2:
                    # Loading phase
                    grf_vertical[frame] = subject_mass * 9.81 * (1.0 + 1.5 * t_norm / 0.2)
                elif t_norm < 0.6:
                    # Mid-stance (valley)
                    valley_factor = 0.8 + 0.4 * np.sin(np.pi * (t_norm - 0.2) / 0.4)
                    grf_vertical[frame] = subject_mass * 9.81 * valley_factor
                else:
                    # Propulsive phase
                    prop_factor = 1.2 + 0.8 * np.sin(np.pi * (t_norm - 0.6) / 0.4)
                    grf_vertical[frame] = subject_mass * 9.81 * prop_factor
                
                # Anterior-posterior GRF (braking then propulsive)
                if t_norm < 0.5:
                    # Braking phase
                    grf_anterior[frame] = -subject_mass * 9.81 * 0.3 * (0.5 - t_norm) / 0.5
                else:
                    # Propulsive phase
                    grf_anterior[frame] = subject_mass * 9.81 * 0.2 * (t_norm - 0.5) / 0.5
        
        return {
            'GRF_vertical': grf_vertical,
            'GRF_anterior': grf_anterior, 
            'GRF_lateral': grf_lateral,
            'in_contact': in_stance,
            'stance_frames': np.sum(in_stance),
            'stance_duration': np.sum(in_stance) * dt
        }
    
    def estimate_grf_for_session(self,
                               session_id: str,
                               joints: Optional[List[str]] = None,
                               normalized: bool = False,
                               method: str = 'simplified') -> Dict[str, np.ndarray]:
        """
        Estimate GRF for a complete session.
        
        Args:
            session_id: Session identifier
            joints: List of joints to include
            normalized: Whether to use normalized data
            method: 'simplified' or 'biorbd' (if available)
            
        Returns:
            Dictionary with estimated GRF data
        """
        # Load kinematic data
        kinematics = self.get_joint_kinematics(session_id, joints, normalized)
        
        if method == 'biorbd' and self.model and BIORBD_AVAILABLE:
            return self._estimate_grf_biorbd(kinematics)
        else:
            # Use simplified method
            if 'L_ankle' in kinematics:
                ankle_data = kinematics['L_ankle']
                ankle_vel_cols = [col for col in ankle_data.columns if '_vel_deg' in col]
                if ankle_vel_cols:
                    # Create velocity DataFrame from combined data
                    vel_data = ankle_data[ankle_vel_cols].copy()
                    vel_data.columns = [col.replace('_vel_deg', '_deg') for col in vel_data.columns]
                    
                    result = self.estimate_grf_simplified(
                        joint_angles=ankle_data[['TimeIndex', 'X_deg', 'Y_deg', 'Z_deg']],
                        joint_velocities=vel_data,
                        subject_mass=70.0
                    )
                    
                    return {'left_foot': result}
            
        return {}
    
    def _estimate_grf_biorbd(self, kinematics: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Estimate GRF using biorbd inverse dynamics (if available).
        
        Args:
            kinematics: Dictionary of joint kinematics data
            
        Returns:
            Dictionary with estimated GRF data
        """
        if not self.model or not BIORBD_AVAILABLE:
            raise RuntimeError("biorbd model not available")
        
        # Convert to biorbd format
        Q, Qdot, Qddot = self.convert_to_biorbd_format(kinematics)
        
        # Estimate contact forces
        contact_forces = self.estimate_contact_forces(Q, Qdot, Qddot)
        
        # Extract GRF components
        grf_data = {}
        for contact_name, forces in contact_forces.items():
            grf_data[contact_name] = {
                'GRF_x': forces['force'][0, :],
                'GRF_y': forces['force'][1, :], 
                'GRF_z': forces['force'][2, :],
                'moment_x': forces['moment'][0, :],
                'moment_y': forces['moment'][1, :],
                'moment_z': forces['moment'][2, :],
                'in_contact': forces['in_contact']
            }
            
        return grf_data
    
    def calculate_grf_features(self, grf_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate key GRF features for biomechanical analysis.
        
        Args:
            grf_data: Dictionary with GRF time series data
            
        Returns:
            Dictionary with calculated features
        """
        features = {}
        
        for contact_name, data in grf_data.items():
            # Handle both simplified and biorbd output formats
            if isinstance(data, dict):
                if 'in_contact' in data and np.any(data['in_contact']):
                    # Extract vertical GRF during contact
                    if 'GRF_y' in data:
                        vertical_grf = data['GRF_y'][data['in_contact']]
                    elif 'GRF_vertical' in data:
                        vertical_grf = data['GRF_vertical'][data['in_contact']]
                    else:
                        continue
                    
                    if len(vertical_grf) > 0:
                        features[f"{contact_name}_peak_grf"] = np.max(vertical_grf)
                        features[f"{contact_name}_mean_grf"] = np.mean(vertical_grf)
                        features[f"{contact_name}_loading_rate"] = self._calculate_loading_rate(vertical_grf)
                        features[f"{contact_name}_impulse"] = np.trapz(vertical_grf)
                        
                        # Impact peak and propulsive peak
                        peaks = self._find_grf_peaks(vertical_grf)
                        if len(peaks) >= 1:
                            features[f"{contact_name}_impact_peak"] = peaks[0]
                        if len(peaks) >= 2:
                            features[f"{contact_name}_propulsive_peak"] = peaks[1]
                        
        return features
    
    def _calculate_loading_rate(self, vertical_grf: np.ndarray, dt: float = 0.01) -> float:
        """Calculate the loading rate of vertical GRF."""
        if len(vertical_grf) < 2:
            return 0.0
            
        # Find the maximum slope in the initial loading phase
        max_loading_rate = 0.0
        for i in range(1, min(len(vertical_grf), 20)):  # First 20 samples
            rate = (vertical_grf[i] - vertical_grf[i-1]) / dt
            max_loading_rate = max(max_loading_rate, rate)
            
        return max_loading_rate
    
    def _find_grf_peaks(self, vertical_grf: np.ndarray) -> List[float]:
        """Find impact and propulsive peaks in vertical GRF."""
        if len(vertical_grf) < 10:
            return []
            
        # Simple peak finding - would use scipy.signal.find_peaks in practice
        peaks = []
        
        # Find local maxima
        for i in range(1, len(vertical_grf) - 1):
            if (vertical_grf[i] > vertical_grf[i-1] and 
                vertical_grf[i] > vertical_grf[i+1] and
                vertical_grf[i] > 0.5 * np.max(vertical_grf)):
                peaks.append(vertical_grf[i])
                
        return sorted(peaks, reverse=True)[:2]  # Return top 2 peaks


def create_simple_biomechanical_model(output_path: str, 
                                    subject_mass: float = 70.0,
                                    subject_height: float = 1.75) -> str:
    """
    Create a simple biomechanical model based on anthropometric data.
    
    Args:
        output_path: Path where to save the model file
        subject_mass: Subject mass in kg
        subject_height: Subject height in m
        
    Returns:
        Path to the created model file
    """
    # Calculate segment masses and dimensions based on anthropometric data
    # These are based on Winter's anthropometric data
    
    total_mass = subject_mass
    pelvis_mass = 0.142 * total_mass
    thigh_mass = 0.100 * total_mass  
    shank_mass = 0.0465 * total_mass
    foot_mass = 0.0145 * total_mass
    
    # Segment lengths
    thigh_length = 0.245 * subject_height
    shank_length = 0.246 * subject_height
    foot_length = 0.152 * subject_height
    
    model_content = f"""version	4

// Anthropometry based model (Mass: {subject_mass}kg, Height: {subject_height}m)

segment	Pelvis
	translations	xyz
	rotations	xyz
	mass	{pelvis_mass:.3f}
	com	0 0 0
	inertia
		{pelvis_mass * 0.1:.4f}	0.0	0.0
		0.0	{pelvis_mass * 0.1:.4f}	0.0
		0.0	0.0	{pelvis_mass * 0.1:.4f}
endsegment

segment	L_Thigh
	parent	Pelvis
	rotations	xyz
	mass	{thigh_mass:.3f}
	com	0 -{thigh_length/2:.3f} 0
	inertia
		{thigh_mass * 0.1:.4f}	0.0	0.0
		0.0	{thigh_mass * 0.1:.4f}	0.0
		0.0	0.0	{thigh_mass * 0.1:.4f}
endsegment

segment	L_Shank
	parent	L_Thigh
	rotations	xyz
	mass	{shank_mass:.3f}
	com	0 -{shank_length/2:.3f} 0
	inertia
		{shank_mass * 0.1:.4f}	0.0	0.0
		0.0	{shank_mass * 0.1:.4f}	0.0
		0.0	0.0	{shank_mass * 0.1:.4f}
endsegment

segment	L_Foot
	parent	L_Shank
	rotations	xyz
	mass	{foot_mass:.3f}
	com	{foot_length/3:.3f} 0 0
	inertia
		{foot_mass * 0.1:.4f}	0.0	0.0
		0.0	{foot_mass * 0.1:.4f}	0.0
		0.0	0.0	{foot_mass * 0.1:.4f}
endsegment

segment	R_Thigh
	parent	Pelvis
	rotations	xyz
	mass	{thigh_mass:.3f}
	com	0 -{thigh_length/2:.3f} 0
	inertia
		{thigh_mass * 0.1:.4f}	0.0	0.0
		0.0	{thigh_mass * 0.1:.4f}	0.0
		0.0	0.0	{thigh_mass * 0.1:.4f}
endsegment

segment	R_Shank
	parent	R_Thigh
	rotations	xyz
	mass	{shank_mass:.3f}
	com	0 -{shank_length/2:.3f} 0
	inertia
		{shank_mass * 0.1:.4f}	0.0	0.0
		0.0	{shank_mass * 0.1:.4f}	0.0
		0.0	0.0	{shank_mass * 0.1:.4f}
endsegment

segment	R_Foot
	parent	R_Shank
	rotations	xyz
	mass	{foot_mass:.3f}
	com	{foot_length/3:.3f} 0 0
	inertia
		{foot_mass * 0.1:.4f}	0.0	0.0
		0.0	{foot_mass * 0.1:.4f}	0.0
		0.0	0.0	{foot_mass * 0.1:.4f}
endsegment

// JOINT DEFINITIONS
joint	L_Hip
	parent	Pelvis
	child	L_Thigh
	rotations	xyz
endjoint

joint	L_Knee
	parent	L_Thigh
	child	L_Shank
	rotations	xyz
endjoint

joint	L_Ankle
	parent	L_Shank
	child	L_Foot
	rotations	xyz
endjoint

joint	R_Hip
	parent	Pelvis
	child	R_Thigh
	rotations	xyz
endjoint

joint	R_Knee
	parent	R_Thigh
	child	R_Shank
	rotations	xyz
endjoint

joint	R_Ankle
	parent	R_Shank
	child	R_Foot
	rotations	xyz
endjoint

// CONTACT DEFINITIONS
contact	L_heel_contact
	parent	L_Foot
	position	-{foot_length/4:.3f} -0.02 0
	axis	yz
endcontact

contact	L_toe_contact
	parent	L_Foot
	position	{foot_length*2/3:.3f} -0.02 0
	axis	yz
endcontact

contact	R_heel_contact
	parent	R_Foot
	position	-{foot_length/4:.3f} -0.02 0
	axis	yz
endcontact

contact	R_toe_contact
	parent	R_Foot
	position	{foot_length*2/3:.3f} -0.02 0
	axis	yz
endcontact
"""
    
    # Write the model file
    with open(output_path, 'w') as f:
        f.write(model_content)
        
    return output_path 
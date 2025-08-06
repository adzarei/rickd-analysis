#!/usr/bin/env python3
"""Pytest tests for the MatlabDataLoader class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from core import MatlabDataLoader


@pytest.fixture
def dummy_test_data():
    """Create dummy test data structure with all necessary files."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create dummy processing summary
    processing_summary = pd.DataFrame({
        'FileIndex': [1, 2, 3, 4],
        'ID': ['100001_1', '100002_1', '100003_1', '100004_1'],
        'SubjectID': ['100001', '100002', '100003', '100004'],
        'SessionID': ['1', '1', '1', '1'],
        'JsonFile': ['100001_1.json', '100002_1.json', '100003_1.json', '100004_1.json'],
        'ProcessingStatus': ['Success', 'Success', 'Error', 'Success'],
        'ErrorMessage': ['', '', 'Invalid data format', '']
    })
    processing_summary.to_csv(temp_dir / 'processing_summary.csv', index=False)

    # Create dummy discrete variables
    discrete_vars = pd.DataFrame({
        'ID': ['100001_1', '100002_1', '100004_1'],
        'Speed_Output': [2.5, 3.0, 2.8],
        'Label': ['walking', 'running', 'walking'],
        'Hz': [100, 100, 100],
        'stride_length': [1.2, 1.5, 1.3],
        'cadence': [90, 95, 88],
        'step_width': [0.15, 0.18, 0.16]
    })
    discrete_vars.to_csv(temp_dir / 'session_discrete_variables.csv', index=False)

    # Create session folders with dummy data
    successful_sessions = ['100001_1', '100002_1', '100004_1']
    joints = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'pelvis']
    markers = ['L_foot_1', 'R_foot_1', 'L_thigh_2', 'R_thigh_2']

    for session_id in successful_sessions:
        session_dir = temp_dir / session_id
        inputs_dir = session_dir / 'inputs'
        results_dir = session_dir / 'results'

        inputs_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        # Create dummy joint angle files
        for joint in joints:
            # Raw angles
            time_data = np.linspace(0, 10, 1000)
            angles_raw = pd.DataFrame({
                'TimeIndex': time_data,
                'X_deg': np.sin(time_data * 2) * 30 + np.random.normal(0, 2, 1000),
                'Y_deg': np.cos(time_data * 1.5) * 20 + np.random.normal(0, 1.5, 1000),
                'Z_deg': np.sin(time_data * 0.5) * 10 + np.random.normal(0, 1, 1000)
            })
            angles_raw.to_csv(results_dir / f'{joint}_angles.csv', index=False)

            # Normalized angles
            gait_cycle = np.linspace(0, 100, 101)
            angles_norm = pd.DataFrame({
                'PercentGaitCycle': gait_cycle,
                'X_deg': np.sin(gait_cycle * np.pi / 50) * 25,
                'Y_deg': np.cos(gait_cycle * np.pi / 30) * 18,
                'Z_deg': np.sin(gait_cycle * np.pi / 80) * 8
            })
            angles_norm.to_csv(results_dir / f'{joint}_norm_angles.csv', index=False)

            # Raw velocities
            velocities_raw = pd.DataFrame({
                'TimeIndex': time_data,
                'X_deg_per_s': np.cos(time_data * 2) * 60 + np.random.normal(0, 5, 1000),
                'Y_deg_per_s': -np.sin(time_data * 1.5) * 30 + np.random.normal(0, 3, 1000),
                'Z_deg_per_s': np.cos(time_data * 0.5) * 5 + np.random.normal(0, 2, 1000)
            })
            velocities_raw.to_csv(results_dir / f'{joint}_velocities.csv', index=False)

            # Normalized velocities
            velocities_norm = pd.DataFrame({
                'PercentGaitCycle': gait_cycle,
                'X_deg_per_s': np.cos(gait_cycle * np.pi / 50) * 50,
                'Y_deg_per_s': -np.sin(gait_cycle * np.pi / 30) * 25,
                'Z_deg_per_s': np.cos(gait_cycle * np.pi / 80) * 4
            })
            velocities_norm.to_csv(results_dir / f'{joint}_norm_velocities.csv', index=False)

        # Create dummy marker data
        for marker in markers:
            marker_data = pd.DataFrame({
                'TimeIndex': time_data,
                'X_coord': np.sin(time_data * 1.2) * 100 + 500 + np.random.normal(0, 5, 1000),
                'Y_coord': np.cos(time_data * 0.8) * 50 + 300 + np.random.normal(0, 3, 1000),
                'Z_coord': np.sin(time_data * 0.3) * 20 + 150 + np.random.normal(0, 2, 1000)
            })
            marker_data.to_csv(inputs_dir / f'{marker}_marker_data.csv', index=False)

        # Create dummy gait events
        gait_events = pd.DataFrame({
            'EventNumber': [1, 2, 3, 4, 5],
            'EventIndex': [100, 350, 600, 850, 950]
        })
        gait_events.to_csv(results_dir / 'gait_cycle_events.csv', index=False)

        # Create dummy joint centers
        joint_centers = pd.DataFrame({
            'Joint': joints,
            'X_coord': [450, 460, 440, 470, 455],
            'Y_coord': [300, 310, 320, 315, 305],
            'Z_coord': [150, 155, 160, 158, 152]
        })
        joint_centers.to_csv(results_dir / 'joint_centers.csv', index=False)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def matlab_loader(dummy_test_data):
    """Create a MatlabDataLoader instance with dummy test data."""
    return MatlabDataLoader(str(dummy_test_data))


@pytest.fixture
def successful_session_id():
    """Provide a sample successful session ID for testing."""
    return '100001_1'


@pytest.fixture
def failed_session_id():
    """Provide a sample failed session ID for testing."""
    return '100003_1'


@pytest.fixture
def test_joint():
    """Provide a sample joint name for testing."""
    return 'L_ankle'


@pytest.fixture
def test_marker():
    """Provide a sample marker name for testing."""
    return 'L_foot_1'


def test_get_processing_summary(matlab_loader):
    """Test loading processing summary data."""
    summary = matlab_loader.get_processing_summary()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4
    assert 'ProcessingStatus' in summary.columns
    assert 'ID' in summary.columns
    assert 'ErrorMessage' in summary.columns

    successful_count = len(summary[summary['ProcessingStatus'] == 'Success'])
    failed_count = len(summary[summary['ProcessingStatus'] == 'Error'])
    assert successful_count == 3
    assert failed_count == 1

def test_get_discrete_variables(matlab_loader):
    """Test loading discrete variables data."""
    discrete_vars = matlab_loader.get_discrete_variables()

    assert isinstance(discrete_vars, pd.DataFrame)
    assert len(discrete_vars) == 3  # Only successful sessions
    assert 'Speed_Output' in discrete_vars.columns
    assert 'Hz' in discrete_vars.columns
    assert 'Label' in discrete_vars.columns

def test_get_successful_sessions(matlab_loader, failed_session_id):
    """Test getting list of successful sessions."""
    successful_sessions = matlab_loader.get_successful_sessions()

    assert isinstance(successful_sessions, list)
    assert len(successful_sessions) == 3
    assert '100001_1' in successful_sessions
    assert '100002_1' in successful_sessions
    assert '100004_1' in successful_sessions
    assert failed_session_id not in successful_sessions

def test_get_failed_sessions(matlab_loader, failed_session_id):
    """Test getting list of failed sessions."""
    failed_sessions = matlab_loader.get_failed_sessions()

    assert isinstance(failed_sessions, list)
    assert len(failed_sessions) == 1
    assert failed_sessions[0][0] == failed_session_id
    assert 'Invalid data format' in failed_sessions[0][1]

def test_get_session_info(matlab_loader, successful_session_id):
    """Test getting comprehensive session information."""
    session_info = matlab_loader.get_session_info(successful_session_id)

    assert isinstance(session_info, dict)
    assert session_info['session_id'] == successful_session_id
    assert 'available_joints' in session_info
    assert 'available_markers' in session_info
    assert len(session_info['available_joints']) == 5
    assert len(session_info['available_markers']) == 4
    assert session_info['processing_status'] == 'Success'

def test_get_available_joints(matlab_loader, successful_session_id):
    """Test getting available joints for a session."""
    joints = matlab_loader.get_available_joints(successful_session_id)

    assert isinstance(joints, list)
    assert len(joints) == 5
    assert 'L_ankle' in joints
    assert 'R_ankle' in joints
    assert 'pelvis' in joints
    assert joints == sorted(joints)  # Should be sorted

def test_get_available_markers(matlab_loader, successful_session_id):
    """Test getting available markers for a session."""
    markers = matlab_loader.get_available_markers(successful_session_id)

    assert isinstance(markers, list)
    assert len(markers) == 4
    assert 'L_foot_1' in markers
    assert 'R_foot_1' in markers
    assert markers == sorted(markers)  # Should be sorted


def test_get_joint_angles_raw(matlab_loader, successful_session_id, test_joint):
    """Test loading raw joint angle data."""
    angles_raw = matlab_loader.get_joint_angles(successful_session_id, test_joint, normalized=False)

    assert isinstance(angles_raw, pd.DataFrame)
    assert len(angles_raw) == 1000
    assert 'TimeIndex' in angles_raw.columns
    assert 'X_deg' in angles_raw.columns
    assert 'Y_deg' in angles_raw.columns
    assert 'Z_deg' in angles_raw.columns


def test_get_joint_angles_normalized(matlab_loader, successful_session_id, test_joint):
    """Test loading normalized joint angle data."""
    angles_norm = matlab_loader.get_joint_angles(successful_session_id, test_joint, normalized=True)

    assert isinstance(angles_norm, pd.DataFrame)
    assert len(angles_norm) == 101
    assert 'PercentGaitCycle' in angles_norm.columns
    assert 'X_deg' in angles_norm.columns
    assert 'Y_deg' in angles_norm.columns
    assert 'Z_deg' in angles_norm.columns

def test_get_joint_velocities_raw(matlab_loader, successful_session_id, test_joint):
    """Test loading raw joint velocity data."""
    velocities_raw = matlab_loader.get_joint_velocities(successful_session_id, test_joint, normalized=False)

    assert isinstance(velocities_raw, pd.DataFrame)
    assert len(velocities_raw) == 1000
    assert 'TimeIndex' in velocities_raw.columns
    assert 'X_deg_per_s' in velocities_raw.columns
    assert 'Y_deg_per_s' in velocities_raw.columns
    assert 'Z_deg_per_s' in velocities_raw.columns

def test_get_joint_velocities_normalized(matlab_loader, successful_session_id, test_joint):
    """Test loading normalized joint velocity data."""
    velocities_norm = matlab_loader.get_joint_velocities(successful_session_id, test_joint, normalized=True)

    assert isinstance(velocities_norm, pd.DataFrame)
    assert len(velocities_norm) == 101
    assert 'PercentGaitCycle' in velocities_norm.columns
    assert 'X_deg_per_s' in velocities_norm.columns
    assert 'Y_deg_per_s' in velocities_norm.columns
    assert 'Z_deg_per_s' in velocities_norm.columns

def test_get_all_joint_angles(matlab_loader, successful_session_id):
    """Test loading angles for all joints in a session."""
    all_angles = matlab_loader.get_all_joint_angles(successful_session_id, normalized=True)

    assert isinstance(all_angles, dict)
    assert len(all_angles) == 5  # Only normalized files since we specified normalized=True

    all_joints = matlab_loader.get_available_joints(successful_session_id)
    assert sorted(all_joints) == sorted(all_angles.keys())

    for _, data in all_angles.items():
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 101  # Normalized data
        assert 'PercentGaitCycle' in data.columns

def test_get_all_joint_velocities(matlab_loader, successful_session_id):
    """Test loading velocities for all joints in a session."""
    all_velocities = matlab_loader.get_all_joint_velocities(successful_session_id, normalized=False)
    print(all_velocities.keys())

    assert isinstance(all_velocities, dict)
    assert len(all_velocities) == 5

    all_joints = matlab_loader.get_available_joints(successful_session_id)
    assert sorted(all_joints) == sorted(all_velocities.keys())

    for _, data in all_velocities.items():
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000  # Raw data
        assert 'TimeIndex' in data.columns

def test_get_gait_events(matlab_loader, successful_session_id):
    """Test loading gait events data."""
    events = matlab_loader.get_gait_events(successful_session_id)

    assert isinstance(events, pd.DataFrame)
    assert len(events) == 5
    assert 'EventNumber' in events.columns
    assert 'EventIndex' in events.columns

def test_get_joint_centers(matlab_loader, successful_session_id):
    """Test loading joint centers data."""
    centers = matlab_loader.get_joint_centers(successful_session_id)

    assert isinstance(centers, pd.DataFrame)
    assert len(centers) == 5
    assert 'Joint' in centers.columns
    assert 'X_coord' in centers.columns
    assert 'Y_coord' in centers.columns
    assert 'Z_coord' in centers.columns

    # Check that all expected joints are present
    joint_names = centers['Joint'].tolist()
    expected_joints = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'pelvis']
    assert all(joint in joint_names for joint in expected_joints)

def test_get_marker_data(matlab_loader, successful_session_id, test_marker):
    """Test loading marker trajectory data."""
    marker_data = matlab_loader.get_marker_data(successful_session_id, test_marker)

    assert isinstance(marker_data, pd.DataFrame)
    assert len(marker_data) == 1000
    assert 'TimeIndex' in marker_data.columns
    assert 'X_coord' in marker_data.columns
    assert 'Y_coord' in marker_data.columns
    assert 'Z_coord' in marker_data.columns

def test_nonexistent_session_error(matlab_loader):
    """Test error handling for non-existent session."""
    fake_session = "FAKE_SESSION_123"

    with pytest.raises(FileNotFoundError, match="Session folder not found"):
        matlab_loader.get_joint_angles(fake_session, "L_ankle")

def test_nonexistent_joint_error(matlab_loader, successful_session_id):
    """Test error handling for non-existent joint."""
    fake_joint = "FAKE_JOINT"

    with pytest.raises(FileNotFoundError, match="Joint angles file not found"):
        matlab_loader.get_joint_angles(successful_session_id, fake_joint)

def test_nonexistent_marker_error(matlab_loader, successful_session_id):
    """Test error handling for non-existent marker."""
    fake_marker = "FAKE_MARKER"

    with pytest.raises(FileNotFoundError, match="Marker data file not found"):
        matlab_loader.get_marker_data(successful_session_id, fake_marker)

def test_invalid_matlab_output_folder():
    """Test error handling for invalid MATLAB output folder."""
    fake_folder = "/path/that/does/not/exist"

    with pytest.raises(FileNotFoundError, match="MATLAB output folder not found"):
        MatlabDataLoader(fake_folder)

def test_processing_summary_caching(matlab_loader):
    """Test that processing summary is cached properly."""
    # First call loads from file
    summary1 = matlab_loader.get_processing_summary()

    # Second call should use cache
    summary2 = matlab_loader.get_processing_summary()

    pd.testing.assert_frame_equal(summary1, summary2)

    # Test cache refresh
    summary3 = matlab_loader.get_processing_summary(refresh_cache=True)
    pd.testing.assert_frame_equal(summary1, summary3)

def test_discrete_variables_caching(matlab_loader):
    """Test that discrete variables are cached properly."""
    # First call loads from file
    vars1 = matlab_loader.get_discrete_variables()

    # Second call should use cache
    vars2 = matlab_loader.get_discrete_variables()

    pd.testing.assert_frame_equal(vars1, vars2)

    # Test cache refresh
    vars3 = matlab_loader.get_discrete_variables(refresh_cache=True)
    pd.testing.assert_frame_equal(vars1, vars3)

def test_full_session_data_workflow(matlab_loader, successful_session_id):
    """Test complete workflow of loading all data for a session."""
    # Get session info
    session_info = matlab_loader.get_session_info(successful_session_id)

    # Load all joint data
    all_angles = matlab_loader.get_all_joint_angles(successful_session_id, normalized=True)
    all_velocities = matlab_loader.get_all_joint_velocities(successful_session_id, normalized=False)

    # Load additional data
    events = matlab_loader.get_gait_events(successful_session_id)
    centers = matlab_loader.get_joint_centers(successful_session_id)

    # Verify consistency
    assert len(session_info['available_joints']) == len(all_angles)
    # Note: all_velocities includes both raw and normalized files due to glob pattern
    raw_velocities = {k: v for k, v in all_velocities.items() if 'TimeIndex' in v.columns}
    assert len(all_angles) == len(raw_velocities)
    assert len(centers) == len(session_info['available_joints'])

    # Verify all joints have consistent data
    for joint in session_info['available_joints']:
        assert joint in all_angles
        assert joint in raw_velocities

        angles_data = all_angles[joint]
        velocities_data = raw_velocities[joint]

        assert len(angles_data) == 101  # Normalized
        assert len(velocities_data) == 1000  # Raw

def test_multiple_sessions_consistency(matlab_loader):
    """Test that multiple sessions have consistent data structure."""
    successful_sessions = matlab_loader.get_successful_sessions()

    for session_id in successful_sessions[:2]:  # Test first 2 sessions
        session_info = matlab_loader.get_session_info(session_id)

        # Each session should have the same joints and markers
        assert len(session_info['available_joints']) == 5
        assert len(session_info['available_markers']) == 4

        # Load one joint from each session to verify structure
        first_joint = session_info['available_joints'][0]
        angles = matlab_loader.get_joint_angles(session_id, first_joint, normalized=True)

        assert len(angles) == 101
        assert list(angles.columns) == ['PercentGaitCycle', 'X_deg', 'Y_deg', 'Z_deg']

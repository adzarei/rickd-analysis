"""Unit tests for the TimeSeries methods in MatlabDataLoader."""

import pytest
import warnings
from core.matlab_data_loader import MatlabDataLoader
import kineticstoolkit as ktk


class TestMatlabDataLoaderTimeSeries:
    """Test class for MatlabDataLoader TimeSeries methods."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures before running tests."""
        cls.loader = MatlabDataLoader()
        successful_sessions = cls.loader.get_successful_sessions()
        if not successful_sessions:
            pytest.skip("No successful sessions found for testing")
        
        cls.test_session = successful_sessions[0]
        cls.available_joints = cls.loader.get_available_joints(cls.test_session)
        cls.available_markers = cls.loader.get_available_markers(cls.test_session)
        
        if not cls.available_joints:
            pytest.skip("No joints available for testing")
        if not cls.available_markers:
            pytest.skip("No markers available for testing")
            
        cls.test_joint = cls.available_joints[0]
        cls.test_marker = cls.available_markers[0]

    def test_joint_angles_timeseries_normalized(self):
        """Test loading normalized joint angles as TimeSeries."""
        ts = self.loader.get_joint_angles_timeseries(
            self.test_session, self.test_joint, normalized=True
        )
        
        # Verify it's a TimeSeries object
        assert isinstance(ts, ktk.TimeSeries)
        
        # Verify time range (normalized should be 0-100%)
        assert len(ts.time) > 0
        assert ts.time[0] > 0  # Should start around 1%
        assert ts.time[-1] <= 100  # Should end at 100%
        
        # Verify data structure
        assert len(ts.data.keys()) == 1
        data_key = list(ts.data.keys())[0]
        assert self.test_joint in data_key
        assert "angle" in data_key
        
        # Verify data dimensions (should be N x 3 for X, Y, Z angles)
        data_shape = ts.data[data_key].shape
        assert len(data_shape) == 2
        assert data_shape[1] == 3

    def test_joint_angles_timeseries_raw_with_events(self):
        """Test loading raw joint angles as TimeSeries with events."""
        ts = self.loader.get_joint_angles_timeseries(
            self.test_session, self.test_joint, normalized=False, include_events=True
        )
        
        # Verify it's a TimeSeries object
        assert isinstance(ts, ktk.TimeSeries)
        
        # Verify time range (raw should be frame indices)
        assert len(ts.time) > 0
        assert ts.time[0] >= 1  # Should start at frame 1
        
        # Verify events were added
        assert len(ts.events) > 0
        
        # Verify event types
        event_names = [event.name for event in ts.events]
        expected_events = ['L_TD', 'L_TO', 'R_TD', 'R_TO']
        assert any(event in event_names for event in expected_events)

    def test_joint_velocities_timeseries(self):
        """Test loading joint velocities as TimeSeries."""
        ts = self.loader.get_joint_velocities_timeseries(
            self.test_session, self.test_joint, normalized=False
        )
        
        # Verify it's a TimeSeries object
        assert isinstance(ts, ktk.TimeSeries)
        
        # Verify data structure
        assert len(ts.data.keys()) == 1
        data_key = list(ts.data.keys())[0]
        assert self.test_joint in data_key
        assert "velocity" in data_key
        
        # Verify data dimensions (should be N x 3 for X, Y, Z velocities)
        data_shape = ts.data[data_key].shape
        assert len(data_shape) == 2
        assert data_shape[1] == 3

    def test_marker_data_timeseries_with_events(self):
        """Test loading marker data as TimeSeries with events."""
        ts = self.loader.get_marker_data_timeseries(
            self.test_session, self.test_marker, include_events=True
        )
        
        # Verify it's a TimeSeries object
        assert isinstance(ts, ktk.TimeSeries)
        
        # Verify data structure
        assert len(ts.data.keys()) == 1
        data_key = list(ts.data.keys())[0]
        assert self.test_marker in data_key
        assert "trajectory" in data_key
        
        # Verify data dimensions (should be N x 3 for X, Y, Z coordinates)
        data_shape = ts.data[data_key].shape
        assert len(data_shape) == 2
        assert data_shape[1] == 3
        
        # Verify events were added
        assert len(ts.events) > 0

    def test_all_joint_angles_timeseries(self):
        """Test loading all joint angles as TimeSeries objects."""
        all_angles = self.loader.get_all_joint_angles_timeseries(
            self.test_session, normalized=True
        )
        
        # Verify it's a dictionary
        assert isinstance(all_angles, dict)
        
        # Verify we got data for multiple joints
        assert len(all_angles) > 0
        
        # Verify each entry is a TimeSeries
        for joint, ts in all_angles.items():
            assert isinstance(ts, ktk.TimeSeries)
            assert joint in self.available_joints

    def test_all_joint_velocities_timeseries(self):
        """Test loading all joint velocities as TimeSeries objects."""
        all_velocities = self.loader.get_all_joint_velocities_timeseries(
            self.test_session, normalized=False, include_events=True
        )
        
        # Verify it's a dictionary
        assert isinstance(all_velocities, dict)
        
        # Verify we got data for multiple joints
        assert len(all_velocities) > 0
        
        # Verify each entry is a TimeSeries with events
        for joint, ts in all_velocities.items():
            assert isinstance(ts, ktk.TimeSeries)
            assert joint in self.available_joints
            assert len(ts.events) > 0

    def test_timeseries_consistency(self):
        """Test that TimeSeries objects have consistent data across methods."""
        # Load same joint data using different methods
        ts_angles = self.loader.get_joint_angles_timeseries(
            self.test_session, self.test_joint, normalized=False
        )
        ts_velocities = self.loader.get_joint_velocities_timeseries(
            self.test_session, self.test_joint, normalized=False
        )
        
        # Time arrays should be similar length (velocities might be one less due to differentiation)
        time_diff = abs(len(ts_angles.time) - len(ts_velocities.time))
        assert time_diff <= 1


def test_timeseries_methods_integration():
    """Integration test to verify all TimeSeries methods work together."""
    # Suppress warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        loader = MatlabDataLoader()
        successful_sessions = loader.get_successful_sessions()
        
        test_session = successful_sessions[0]
        
        # Test that we can load different data types for the same session
        try:
            # Get available data
            joints = loader.get_available_joints(test_session)
            markers = loader.get_available_markers(test_session)
            
            if joints:
                # Test angles
                ts_angles = loader.get_joint_angles_timeseries(
                    test_session, joints[0], include_events=True
                )
                assert isinstance(ts_angles, ktk.TimeSeries)
                
                # Test velocities  
                ts_velocities = loader.get_joint_velocities_timeseries(
                    test_session, joints[0], include_events=True
                )
                assert isinstance(ts_velocities, ktk.TimeSeries)
            
            if markers:
                # Test markers
                ts_marker = loader.get_marker_data_timeseries(
                    test_session, markers[0], include_events=True
                )
                assert isinstance(ts_marker, ktk.TimeSeries)
                
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 
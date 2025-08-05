"""
Pytest configuration and common fixtures for kinematics tests
"""

import pytest
import numpy as np
import warnings

# Configure numpy and warnings for tests
np.random.seed(42)  # For reproducible tests
warnings.filterwarnings("ignore", category=UserWarning, module="kineticstoolkit")


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Add unit marker to tests by default
        if "integration" not in [mark.name for mark in item.iter_markers()]:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    from pathlib import Path
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up common test environment"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Suppress specific warnings during tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield


# Performance fixture for timing tests
@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer() 
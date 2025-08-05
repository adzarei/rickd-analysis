# Python Kinematics Module - Conversion Guide

## Overview

This document describes the Python conversion of the MATLAB gait analysis functions from the Running Injury Clinic. The Python module provides equivalent functionality using modern Python libraries and Kinetics Toolkit where possible.

## Converted Functions

### MATLAB → Python Function Mapping

| MATLAB Function | Python Function | Purpose |
|----------------|----------------|---------|
| `gait_kinematics.m` | `gait_kinematics()` | Calculate joint angles using SVD-based pose estimation |
| `pca_td.m` | `PCAEventDetector.detect_touchdown_events()` | PCA-based touchdown event detection |
| `pca_to.m` | `PCAEventDetector.detect_toeoff_events()` | PCA-based toe-off event detection |
| `gait_steps.m` | `gait_steps()` | Complete gait analysis pipeline |
| `processing_code_example.m` | `processing_pipeline()` | Main wrapper function |

## Dependencies

### Required Packages

```python
# Core scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import pdist2
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

# Machine learning
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Biomechanics toolkit
import kineticstoolkit as ktk
```

### Installation

```bash
# Install all dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Usage Examples

### Basic Usage (Direct Function Calls)

```python
from src.core.kinematics import gait_kinematics, gait_steps

# Load your data (replace with actual data loading)
joints = {...}      # Joint center locations
neutral = {...}     # Static trial marker positions  
dynamic = {...}     # Dynamic trial marker positions
hz = 200.0         # Sampling frequency

# Calculate joint kinematics
angles, velocities, jc, R, djc = gait_kinematics(
    joints, neutral, dynamic, hz, plots=True)

# Analyze gait cycles
norm_angles, norm_velocities, events, event, discrete_vars, speed, flags, gait_type = gait_steps(
    neutral, dynamic, angles, velocities, hz, plots=True)

# Access results
print(f"Gait type: {gait_type}")
print(f"Speed: {speed:.2f} m/s")
print(f"Discrete variables: {discrete_vars}")
```

### JSON Processing Pipeline (MATLAB Equivalent)

```python
from src.core.kinematics import processing_pipeline

# Process JSON file (equivalent to MATLAB processing_code_example.m)
results = processing_pipeline('path/to/subject_data.json', plots=True)

# Access results for different gait types
if 'walking' in results:
    walking_data = results['walking']
    walking_angles = walking_data['angles']
    walking_speed = walking_data['speed']

if 'running' in results:
    running_data = results['running']  
    running_angles = running_data['angles']
    running_speed = running_data['speed']
```

### Using Individual Components

```python
from src.core.kinematics import SoderqvistPoseEstimator, PCAEventDetector, GaitClassifier

# SVD-based pose estimation
pose_estimator = SoderqvistPoseEstimator()
pose_estimator.setup_reference_configuration(neutral_markers)
rotations = pose_estimator.track_segments(dynamic_markers)

# Event detection
event_detector = PCAEventDetector()
left_td, right_td = event_detector.detect_touchdown_events(angles, hz, 'run')
left_to, right_to = event_detector.detect_toeoff_events(angles, hz, 'run')

# Gait classification
classifier = GaitClassifier()
gait_type = classifier.classify_gait(velocity=2.5, step_rate=180)
```

## Key Features

### 1. Soderqvist SVD-Based Pose Estimation

The Python implementation includes the same SVD-based rigid body tracking algorithm:

```python
# Equivalent to MATLAB's soderkvist algorithm
class SoderqvistPoseEstimator:
    def estimate_pose(self, dynamic_markers, frame_idx):
        # Steps 1-2: Calculate centroids and center point sets
        # Step 3: Cross-correlation matrix
        C = dynamic_centered.T @ reference_centered
        # Step 4: SVD decomposition  
        P, T, Q_T = np.linalg.svd(C)
        # Step 5: Calculate rotation matrix
        R = P @ diag_matrix @ Q.T
        return R
```

### 2. PCA-Based Event Detection

Implements the same PCA models for touchdown and toe-off detection:

```python
# Equivalent to pca_td.m and pca_to.m
detector = PCAEventDetector()
left_events, right_events = detector.detect_touchdown_events(angles, hz, gait_type)
```

### 3. Gait Analysis Pipeline

Complete analysis equivalent to `gait_steps.m`:

- Gait type classification (walk vs run)
- Event detection and validation
- Cycle normalization
- Discrete variable calculation
- Speed and step rate estimation

### 4. Kinetics Toolkit Integration

The module uses Kinetics Toolkit for advanced geometric operations:

```python
# Use KTK for advanced geometric operations
angles = ktk.geometry.get_angles(prox_transforms, dist_transforms)
```

## Data Structures

### Input Data Format

The Python module expects the same data structure as the MATLAB version:

```python
# Joint centers (from static trial)
joints = {
    'L_hip': np.array([x, y, z]),
    'R_hip': np.array([x, y, z]),
    'L_lat_knee': np.array([x, y, z]),
    'L_med_knee': np.array([x, y, z]),
    # ... etc
}

# Neutral markers (static trial)
neutral = {
    'L_foot': np.array([[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]),  # 3 markers
    'R_foot': np.array([[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]),
    # ... etc
}

# Dynamic markers (motion trial)
dynamic = {
    'L_foot': np.array([n_frames, n_markers*3]),  # Flattened XYZ data
    'R_foot': np.array([n_frames, n_markers*3]),
    # ... etc  
}
```

### Output Data Format

```python
# Joint angles (equivalent to MATLAB angles struct)
angles = {
    'L_hip': np.array([n_frames, 3]),    # [flexion/extension, ab/adduction, rotation]
    'R_hip': np.array([n_frames, 3]),
    'L_knee': np.array([n_frames, 3]),
    # ... etc
}

# Discrete variables (equivalent to MATLAB DISCRETE_VARIABLES)
discrete_vars = {
    'speed': 2.1,                    # m/s
    'step_rate': 120,               # steps/min
    'gait_type': 'run',
    'L_knee_peak_flexion': 45.2,   # degrees
    'L_knee_range_of_motion': 38.1, # degrees
    # ... etc
}
```

## Coordinate Systems

The Python module maintains the same coordinate system conventions as the MATLAB version:

**Lab Coordinate System:**
- X: Points to subject's right  
- Y: Points vertically upwards
- Z: Points opposite to walking direction

**Segment Coordinate System:**
- X: Anterior (Ab/Adduction)
- Y: Vertically upwards (Axial rotation)  
- Z: Points to subject's right (Flexion/Extension)

## Algorithm Equivalence

### 1. Joint Angle Calculation

Both MATLAB and Python versions use:
- SVD-based pose estimation (Soderqvist method)
- Euler angle extraction (XYZ sequence)
- Same rotation matrix calculations

### 2. Event Detection

Both versions implement:
- PCA-based touchdown detection (Osis et al. 2014)
- Foot-forward/foot-back fallback detection
- Same peak detection parameters
- Identical filtering approaches

### 3. Gait Classification  

Both versions use:
- Linear Discriminant Analysis (LDA)
- Same features: velocity and step rate
- Identical classification thresholds

## Performance Considerations

### Speed Optimizations

```python
# Use vectorized operations where possible
angles_vectorized = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))

# Parallel processing for multiple subjects
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(process_subject, subject_files)
```

### Memory Management

```python
# Pre-allocate arrays for large datasets
n_frames = len(dynamic_data)
angles = np.zeros((n_frames, 3))

# Use generators for large file processing
def process_files_generator(file_list):
    for file_path in file_list:
        yield processing_pipeline(file_path)
```

## Testing and Validation

### Running Tests with Pytest

The project uses pytest for comprehensive testing with proper fixtures, markers, and coverage reporting.

```bash
# Install development dependencies (includes pytest)
poetry install

# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src/core --cov-report=term-missing

# Run only unit tests (fast)
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run tests verbosely
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_kinematics.py

# Run specific test class or function
poetry run pytest tests/test_kinematics.py::TestGaitKinematics::test_gait_kinematics_outputs
```

### Test Structure

The test suite is organized into:

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows  
- **Parametrized tests**: Test with different configurations
- **Performance tests**: Marked as `slow` for optional execution

### Expected Output

```bash
$ poetry run pytest tests/ --tb=short -q
.............s........s                                                                              [100%]
21 passed, 2 skipped, 1 warning in 2.14s

# With coverage:
$ poetry run pytest tests/ --cov=src/core --cov-report=term-missing
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
src/core/kinematics.py       420    193    54%   [line numbers]
Total: 54% coverage
```

### Test Features

- **Fixtures**: Reusable test data for consistent testing
- **Markers**: Categorize tests (unit, integration, slow)  
- **Coverage**: Track code coverage and identify untested areas
- **Parametrized**: Test multiple scenarios automatically
- **Reproducible**: Fixed random seeds for consistent results

## Troubleshooting

### Common Issues

1. **Data Format Issues**
   ```python
   # Ensure data is in correct format
   if isinstance(data, list):
       data = np.array(data)
   ```

2. **Coordinate System Differences**
   ```python
   # Check coordinate system orientation
   if coordinate_system == 'bonita':
       # Apply coordinate transformation
       data = transform_coordinates(data)
   ```

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate input data
def validate_input_data(joints, neutral, dynamic):
    assert all(isinstance(v, np.ndarray) for v in joints.values())
    assert all(v.shape == (3,) for v in joints.values())
    # ... additional checks
```

## Migration from MATLAB

### Step-by-Step Migration

1. **Install Python environment**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   cd rickd-analysis
   poetry install
   poetry shell
   ```

2. **Convert data format**
   ```python
   # If migrating from .mat files
   from scipy.io import loadmat
   matlab_data = loadmat('subject_data.mat')
   
   # Convert to Python format
   python_data = convert_matlab_structure(matlab_data)
   ```

3. **Update analysis scripts**
   ```python
   # Old MATLAB code:
   # [angles, velocities, jc, R, djc] = gait_kinematics(joints, neutral, dynamic, hz, plots);
   
   # New Python code:
   angles, velocities, jc, R, djc = gait_kinematics(joints, neutral, dynamic, hz, plots=True)
   ```

4. **Verify results**
   ```python
   # Compare outputs between MATLAB and Python
   matlab_results = load_matlab_results('matlab_output.mat')
   python_results = {'angles': angles, 'velocities': velocities}
   compare_results(matlab_results, python_results)
   ```

## Future Enhancements

### Planned Features

1. **Enhanced Kinetics Toolkit Integration**
   - Full 3D visualization
   - Advanced filtering options
   - Improved coordinate transformations

2. **Machine Learning Extensions**
   - Automated injury risk assessment
   - Movement pattern clustering
   - Predictive modeling

3. **Performance Optimizations**
   - GPU acceleration for large datasets
   - Parallel processing improvements
   - Memory usage optimization

4. **Additional File Formats**
   - C3D file support
   - Direct motion capture system integration
   - Real-time processing capabilities

## References

1. Osis et al. (2014). A novel method to evaluate initial ground contact event timing
2. Soderqvist & Wedin (1993). Determining the movements of the skeleton
3. Running Injury Clinic - Original MATLAB Implementation
4. [Kinetics Toolkit Documentation](https://kineticstoolkit.uqam.ca/doc/index.html)

## Support

For questions or issues with the Python conversion:

1. Check the test suite output for validation
2. Review the original MATLAB documentation
3. Consult the Kinetics Toolkit documentation for advanced features
4. Submit issues with sample data and expected outputs 
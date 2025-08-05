# Tests for Kinematics Module

This directory contains comprehensive tests for the Python kinematics module converted from MATLAB.

## Test Structure

```
tests/
├── __init__.py          # Tests package initialization
├── conftest.py          # Pytest configuration and shared fixtures
├── test_kinematics.py   # Main test suite for kinematics module
└── README.md           # This file
```

## Running Tests

### Basic Usage

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=src/core --cov-report=term-missing
```

### Selective Testing

```bash
# Run only unit tests (fast)
poetry run pytest -m unit

# Run only integration tests  
poetry run pytest -m integration

# Skip slow tests
poetry run pytest -m "not slow"

# Run specific test class
poetry run pytest tests/test_kinematics.py::TestGaitKinematics

# Run specific test method
poetry run pytest tests/test_kinematics.py::TestGaitKinematics::test_gait_kinematics_outputs
```

## Test Categories

### Unit Tests (Default)
- Test individual functions and classes
- Fast execution
- Isolated components
- No external dependencies

### Integration Tests
- Test complete workflows
- Test component interactions
- Marked with `@pytest.mark.integration`

### Performance Tests
- Test with large datasets
- Marked with `@pytest.mark.slow`
- Skipped by default (run with `--runslow`)

## Test Classes

### `TestSoderqvistPoseEstimator`
Tests for SVD-based pose estimation:
- Initialization and setup
- Reference configuration
- Single frame pose estimation
- Full motion tracking
- Rotation matrix validation

### `TestPCAEventDetector`
Tests for PCA-based event detection:
- Initialization
- Touchdown detection
- Toe-off detection
- Event structure validation

### `TestGaitClassifier`
Tests for gait classification:
- Walk vs run classification
- Threshold-based decisions
- Velocity-based classification

### `TestGaitKinematics`
Tests for main kinematics function:
- Output structure validation
- Angle range verification
- Joint calculation accuracy

### `TestGaitSteps`
Tests for gait step analysis:
- Step detection
- Cycle normalization
- Discrete variable calculation

### `TestProcessingPipeline`
Tests for JSON processing pipeline:
- File format compatibility
- Complete workflow testing
- Error handling

### `TestIntegrationTests`
End-to-end workflow tests:
- Complete analysis pipeline
- Algorithm consistency
- Performance validation

## Fixtures

### Data Fixtures
- `sample_joints`: Joint center locations
- `sample_neutral`: Static trial marker data
- `sample_dynamic`: Dynamic trial marker data with realistic motion
- `sample_hz`: Sampling frequency
- `gait_data`: Complete GaitData object

### Utility Fixtures
- `test_data_dir`: Path to test data directory
- `setup_test_environment`: Common test setup (auto-used)
- `benchmark_timer`: Performance timing utility

## Parametrized Tests

Tests that run with multiple parameter sets:
- Different sampling rates (100, 200, 250 Hz)
- Different gait types (walk, run)
- Various data configurations

## Coverage

Current test coverage:
- **54%** of kinematics module
- Focus on core algorithms
- Major functions tested
- Error handling verified

### Coverage Report
```bash
poetry run pytest --cov=src/core --cov-report=html
# Opens htmlcov/index.html for detailed coverage
```

## Test Data

Tests use synthetic data that mimics real gait analysis data:
- Realistic joint positions
- Sinusoidal motion patterns
- Proper segment relationships
- Noise simulation

## Debugging Tests

### Verbose Output
```bash
poetry run pytest -v -s
```

### Stop on First Failure
```bash
poetry run pytest -x
```

### Run Specific Failing Test
```bash
poetry run pytest tests/test_kinematics.py::test_name --tb=long
```

### Print Statements in Tests
```bash
poetry run pytest -s  # Don't capture stdout
```

## Continuous Integration

Tests are designed to run in CI environments:
- No external dependencies (besides KTK)
- Reproducible with fixed random seeds
- Fast execution (< 5 seconds)
- Clear failure reporting

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for individual functions
2. **Add integration tests** for workflows
3. **Use appropriate fixtures** for test data
4. **Mark tests appropriately** (unit/integration/slow)
5. **Update coverage** and verify new code is tested

### Example Test

```python
def test_new_function(sample_joints):
    """Test new functionality"""
    result = new_function(sample_joints)
    assert isinstance(result, dict)
    assert len(result) > 0
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `poetry install` has been run
2. **Data format errors**: Expected with synthetic test data
3. **Kinetics Toolkit warnings**: Normal and suppressed in tests
4. **Slow performance**: Use `-m "not slow"` to skip performance tests

### Getting Help

- Check test output with `-v` for verbose information
- Use `--tb=long` for detailed tracebacks  
- Review fixtures in `conftest.py`
- Check the main documentation in `docs/` 
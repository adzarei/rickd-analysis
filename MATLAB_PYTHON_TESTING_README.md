# MATLAB vs Python Gait Kinematics Testing

This document explains how to test the Python implementation of the gait kinematics pipeline against the original MATLAB implementation.

## Overview

The testing process involves:
1. Running the updated MATLAB script to process one JSON file and save outputs
2. Running the Python test script to compare MATLAB and Python results
3. Reviewing comparison plots and numerical differences

## Files Created

### 1. `processing_code_example_with_outputs.m`
Updated MATLAB script that:
- Processes ONE JSON file (first file found)
- Saves all outputs from `gait_kinematics` and `gait_steps` functions
- Creates organized .mat files for easy loading in Python
- Provides detailed logging and error handling

### 2. `test_matlab_vs_python.py`
Python script that:
- Loads MATLAB results from .mat files
- Runs the Python implementation on the same JSON file
- Compares outputs numerically with configurable tolerance
- Creates comparison plots for visual inspection
- Reports detailed test results

## Usage Instructions

### Step 1: Run MATLAB Script

1. Open MATLAB
2. Navigate to the supplemental materials folder:
   ```matlab
   cd('/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code')
   ```

3. Run the updated script:
   ```matlab
   processing_code_example_with_outputs
   ```

This will:
- Process the first JSON file found in the source_data directory
- Create a `matlab_outputs` folder in the dataset directory
- Save several .mat files with organized results

### Step 2: Run Python Comparison

1. Navigate to the rickd-analysis directory:
   ```bash
   cd /Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis
   ```

2. Ensure you have the required Python packages:
   ```bash
   pip install numpy scipy matplotlib
   ```

3. Run the comparison script:
   ```bash
   python test_matlab_vs_python.py
   ```

## Output Files

### MATLAB Outputs (saved in `matlab_outputs/` directory):
- `{subject}_matlab_results.mat` - Complete results structure
- `{subject}_inputs.mat` - Input data for Python testing
- `{subject}_walking_results.mat` - Walking-specific results (if available)
- `{subject}_running_results.mat` - Running-specific results (if available)

### Python Test Outputs:
- Console output with detailed comparison results
- Comparison plots in `matlab_outputs/comparison_plots/`
- Pass/fail status for each tested component

## What Gets Tested

### Gait Kinematics Outputs:
- Joint angles (L_ankle, R_ankle, L_knee, R_knee, L_hip, R_hip)
- Joint velocities (same joints)
- Joint centers (pelvis, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle)
- Rotation matrices
- Distance to joint centers (djc)

### Data Types:
- Walking data (if available in JSON)
- Running data (if available in JSON)

## Interpreting Results

### ✅ Success Indicators:
- "Arrays match" messages with very small differences (< 1e-10)
- "ALL TESTS PASSED!" summary message
- Overlapping lines in comparison plots

### ❌ Failure Indicators:
- "Arrays differ" messages with large differences
- "Shape mismatch" errors
- "SOME TESTS FAILED" summary message
- Visible differences in comparison plots

### Tolerance Settings:
- Default tolerance: 1e-10 (very strict)
- Can be adjusted in `compare_arrays()` function if needed
- Small differences may be acceptable due to floating-point precision

## Troubleshooting

### Common Issues:

1. **No MATLAB results found:**
   - Ensure the MATLAB script ran successfully
   - Check that .mat files were created in the matlab_outputs directory
   - Verify file paths are correct

2. **Python import errors:**
   - Ensure you're running from the rickd-analysis directory
   - Check that the src/core/kinematics.py module exists
   - Install required dependencies

3. **Shape mismatches:**
   - May indicate fundamental differences in data processing
   - Check coordinate system transformations
   - Verify input data formatting

4. **Large numerical differences:**
   - Review algorithm implementations
   - Check matrix operations and calculations
   - Verify unit conversions (radians vs degrees)

## Customization

### To test different subjects:
Modify the MATLAB script to process a specific file:
```matlab
% Change this line in the MATLAB script:
i = 1;  % Change to desired file index
```

### To adjust comparison tolerance:
Modify the Python script:
```python
# In compare_arrays function:
tolerance: float = 1e-10  # Adjust as needed
```

### To test additional outputs:
Add new comparison functions in the Python script following the pattern of `test_gait_kinematics_outputs()`.

## Expected Workflow

1. **First Run:** Establish baseline with known working data
2. **Development:** Use for regression testing when modifying Python code
3. **Validation:** Verify Python implementation matches MATLAB exactly
4. **Documentation:** Generate plots and reports for validation documentation

## Notes

- The scripts are currently configured for the specific file paths in the development environment
- Paths can be modified at the top of each script for different environments
- The MATLAB script processes only one file for efficiency during development
- Results are saved in MATLAB v7.3 format for better Python compatibility 
# MATLAB Gait Processing with Enhanced CSV Export

This document explains how to use the enhanced MATLAB processing script that provides comprehensive data export, robust error handling, and organized output structure for easy analysis in Python.

## Overview

The enhanced processing pipeline provides:
1. **Flexible processing modes**: Single file or batch processing of all files
2. **Comprehensive data export**: Complete kinematics, gait analysis, and input data
3. **Organized output structure**: Separate folders for inputs and results with clear hierarchy
4. **Robust error handling**: Column validation, NaN padding, and detailed warnings
5. **Progress tracking**: Visual waitbar with real-time status updates
6. **Smart data filtering**: Only exports meaningful discrete variables (non-empty/non-zero)
7. **Python integration**: CSV format optimized for pandas dataframes

## Files Structure

```
src/supplemental_material/Code/
├── processing_code_example_with_outputs.m    # Enhanced MATLAB processing script
├── gait_kinematics.m                         # Kinematics analysis function
├── gait_steps.m                              # Gait steps analysis function
└── other_matlab_functions.m                  # Supporting functions
```

## Configuration

### MATLAB Script Configuration

Edit the configuration section in `processing_code_example_with_outputs.m`:

```matlab
%% Configuration

% Set processing mode: 'single' or 'all'
PROCESSING_MODE = 'single';  % Change to 'all' to process all files

% Set which file to process if using single mode (1 = first file, 2 = second file, etc.)
SINGLE_FILE_INDEX = 1;

% Path configurations (update these for your system)
code_folder = '/path/to/rickd-analysis/src/supplemental_material/Code';
data_folder = '/path/to/data/Running Injury Clinic Kinematic Dataset/source_data';
output_folder = '/path/to/data/Running Injury Clinic Kinematic Dataset/processed_data/matlab_output';
```

## Output Directory Structure

The script creates a comprehensive, organized output structure:

```
matlab_output/
├── processing_summary.csv              # Processing status for all files
├── discrete_variables.csv              # Combined discrete variables (non-empty only)
└── {ID}/                               # Individual subject/session folders
    ├── inputs/                         # Raw input data
    │   ├── neutral_joint_markers.csv   # Neutral joint positions
    │   └── joint_markers.csv           # Joint marker positions
    └── results/                        # Processed gait analysis results
        ├── {joint}_angles.csv          # Joint angles over time
        ├── {joint}_velocities.csv      # Joint velocities over time
        ├── {joint}_norm_angles.csv     # Normalized joint angles (% gait cycle)
        ├── {joint}_norm_velocities.csv # Normalized joint velocities (% gait cycle)
        ├── joint_centers.csv           # All joint center coordinates
        ├── djc.csv                     # Joint center derivatives
        └── event.csv                   # Gait events timing
```

### File Naming Convention

- **ID Format**: `{subject_id}_{session_id}` (e.g., "100001_20110531T161051")
- **Joint Files**: One file per joint (e.g., "ankle_l_angles.csv", "knee_r_velocities.csv")
- **Combined Files**: Multiple joints in single file with joint name column

## Data Export Details

### Root Level Files

#### `processing_summary.csv`
Contains processing status for each file:
- **ID**: Subject and session identifier
- **SubjectID**: Subject identifier only
- **SessionID**: Session identifier only  
- **JsonFile**: Full path to source JSON file
- **ProcessingStatus**: "Success" or "Error"
- **ErrorMessage**: Details if processing failed

#### `discrete_variables.csv`
Combined discrete variables from all successfully processed files:
- **ID**: Subject and session identifier
- **Speed_Output**: Running speed output
- **Label**: Gait type label (typically "run")
- **Hz**: Sampling frequency
- **{Variable}_Left/Right**: 77 discrete gait variables (only non-empty ones included)

### Individual Subject Folders

#### Input Data (`inputs/` subfolder)

**`neutral_joint_markers.csv`**: Neutral stance joint positions
```csv
Joint,X_coord,Y_coord,Z_coord
pelvis,1.234,5.678,9.012
hip_l,2.345,6.789,0.123
hip_r,3.456,7.890,1.234
...
```

**`joint_markers.csv`**: Dynamic joint marker positions
```csv
Joint,X_coord,Y_coord,Z_coord
ankle_l,1.111,2.222,3.333
ankle_r,4.444,5.555,6.666
knee_l,7.777,8.888,9.999
...
```

#### Results Data (`results/` subfolder)

**Joint-specific files** (one per joint):
- **`{joint}_angles.csv`**: TimeIndex, X_deg, Y_deg, Z_deg
- **`{joint}_velocities.csv`**: TimeIndex, X_deg_per_s, Y_deg_per_s, Z_deg_per_s
- **`{joint}_norm_angles.csv`**: PercentGaitCycle, X_deg, Y_deg, Z_deg
- **`{joint}_norm_velocities.csv`**: PercentGaitCycle, X_deg_per_s, Y_deg_per_s, Z_deg_per_s

**Combined files**:
- **`joint_centers.csv`**: Joint, X_coord, Y_coord, Z_coord
- **`djc.csv`**: Joint, X_velocity, Y_velocity, Z_velocity  
- **`event.csv`**: EventNumber, EventIndex

## Key Features

### 1. Robust Data Handling
- **Column validation**: Automatically detects missing coordinate dimensions
- **NaN padding**: Fills missing coordinates with NaN values
- **Warning system**: Detailed warnings for data quality issues
- **Error recovery**: Continues processing even if individual files fail

### 2. Progress Tracking
- **Visual waitbar**: Real-time progress indication
- **Status messages**: Current file being processed
- **Completion summary**: Final statistics on successful/failed processing

### 3. Smart Data Filtering
- **Non-empty variables only**: Discrete variables with all zeros or NaNs are excluded
- **Dynamic headers**: Column structure adapts based on available data
- **Quality indicators**: Clear identification of missing or placeholder data

### 4. Error Handling Examples

```matlab
Warning: Joint "ankle_r" angles data has only 1 columns instead of expected 3. Padding with NaN.
Warning: Joint "knee_l" djc data has only 2 columns instead of expected 3. Padding with NaN.
Warning: Neutral joint "hip_l" data has only 2 columns instead of expected 3. Padding with NaN.
```

## Usage Instructions

### 1. Single File Processing
```matlab
% Set configuration
PROCESSING_MODE = 'single';
SINGLE_FILE_INDEX = 1;  % Process first file

% Run the script
processing_code_example_with_outputs
```

### 2. Batch Processing
```matlab
% Set configuration  
PROCESSING_MODE = 'all';

% Run the script (will process all JSON files)
processing_code_example_with_outputs
```

### 3. Monitor Progress
- Waitbar window shows real-time progress
- Console output provides detailed processing information
- Warnings appear for any data quality issues

## Python Integration

The CSV output format is optimized for Python analysis:

```python
import pandas as pd
import os

# Load processing summary
summary = pd.read_csv('processing_summary.csv')

# Load combined discrete variables
discrete_vars = pd.read_csv('discrete_variables.csv')

# Load individual subject data
subject_id = "100001_20110531T161051"
inputs_path = f"{subject_id}/inputs/"
results_path = f"{subject_id}/results/"

# Load input data
neutral_markers = pd.read_csv(f"{inputs_path}/neutral_joint_markers.csv")
joint_markers = pd.read_csv(f"{inputs_path}/joint_markers.csv")

# Load joint-specific results
ankle_angles = pd.read_csv(f"{results_path}/ankle_l_angles.csv")
knee_velocities = pd.read_csv(f"{results_path}/knee_r_velocities.csv")

# Load combined results
joint_centers = pd.read_csv(f"{results_path}/joint_centers.csv")
gait_events = pd.read_csv(f"{results_path}/event.csv")
```

## Output Statistics

After processing, the script provides:
- **Total files processed**: Count of attempted files
- **Successful**: Count of successfully processed files  
- **Errors**: Count of failed files with error details
- **File locations**: Paths to all output directories

## Troubleshooting

### Common Issues

1. **Path Configuration**: Ensure all folder paths are correct for your system
2. **MATLAB Path**: Add the code folder to MATLAB path
3. **Memory Issues**: For large datasets, process in smaller batches
4. **Missing Functions**: Ensure `gait_kinematics.m` and `gait_steps.m` are in the code folder

### Data Quality Warnings

- **Missing coordinates**: Joint data with < 3 dimensions will be padded with NaN
- **Empty discrete variables**: Variables with all zeros/NaNs are automatically excluded
- **Processing errors**: Check `processing_summary.csv` for detailed error messages

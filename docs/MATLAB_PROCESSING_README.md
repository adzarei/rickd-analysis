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
7. **Enhanced file naming**: Descriptive filenames for better data organization
8. **Python integration**: CSV format optimized for pandas dataframes

## Files Structure

```
src/
├── scripts/
│   └── processing_code_example_with_outputs.m     # Main MATLAB processing script (entry point)
└── supplemental_material/
    └── Code/
        ├── pca_td.m                              # PCA function for touchdown (TD) events
        ├── pca_to.m                              # PCA function for toe-off (TO) events
        ├── gait_kinematics.m                     # Kinematics analysis function
        └── gait_steps.m                          # Gait steps analysis function
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
├── processing_summary.csv                    # Processing status for all files
├── session_discrete_variables.csv            # Combined discrete variables (non-empty only)
└── {ID}/                                     # Individual subject/session folders
    ├── {ID}_matlab_results.mat               # Complete MATLAB results structure
    ├── inputs/                               # Raw input data
    │   ├── neutral_joint_marker_centers.csv  # Neutral joint positions
    │   ├── joint_marker_centers.csv          # Joint marker positions
    │   └── {marker}_marker_data.csv          # Running marker trajectories (one per marker)
    └── results/                              # Processed gait analysis results
        ├── {joint}_angles.csv                # Joint angles over time
        ├── {joint}_velocities.csv            # Joint velocities over time
        ├── {joint}_norm_angles.csv           # Normalized joint angles (% gait cycle)
        ├── {joint}_norm_velocities.csv       # Normalized joint velocities (% gait cycle)
        ├── joint_centers.csv                 # All joint center coordinates
        ├── distance_to_joint_centers.csv     # Joint center derivatives
        └── gait_cycle_events.csv             # Gait events timing
```

### File Naming Convention

- **ID Format**: `{subject_id}_{session_id}` (e.g., "100001_20110531T161051")
- **Joint Files**: One file per joint (e.g., "ankle_l_angles.csv", "knee_r_velocities.csv")
- **Combined Files**: Multiple joints in single file with joint name column
- **Enhanced Naming**: Descriptive filenames for better organization:
  - `neutral_joint_marker_centers.csv` (instead of `neutral_joint_markers.csv`)
  - `joint_marker_centers.csv` (instead of `joint_markers.csv`)
  - `{marker}_marker_data.csv` (instead of `{marker}_running.csv`)
  - `distance_to_joint_centers.csv` (instead of `djc.csv`)
  - `gait_cycle_events.csv` (instead of `event.csv`)
  - `session_discrete_variables.csv` (instead of `discrete_variables.csv`)

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

#### `session_discrete_variables.csv`
Combined discrete variables from all successfully processed files:
- **ID**: Subject and session identifier
- **Speed_Output**: Running speed output
- **Label**: Gait type label (typically "run")
- **Hz**: Sampling frequency
- **{Variable}_Left/Right**: 77 discrete gait variables (only non-empty ones included)

### Individual Subject Folders

Each subject/session folder contains:

#### MATLAB Results File

**`{ID}_matlab_results.mat`**: Complete MATLAB data structure
- Contains all input data, processing results, and metadata
- Can be loaded directly in MATLAB for further analysis
- Includes both raw inputs and processed outputs in original MATLAB format

#### Input Data (`inputs/` subfolder)

**`neutral_joint_marker_centers.csv`**: Neutral stance joint positions
```csv
Joint,X_coord,Y_coord,Z_coord
pelvis,1.234,5.678,9.012
hip_l,2.345,6.789,0.123
hip_r,3.456,7.890,1.234
...
```

**`joint_marker_centers.csv`**: Dynamic joint marker positions
```csv
Joint,X_coord,Y_coord,Z_coord
ankle_l,1.111,2.222,3.333
ankle_r,4.444,5.555,6.666
knee_l,7.777,8.888,9.999
...
```

**`{marker}_marker_data.csv`**: Running marker trajectories (one file per marker)
```csv
TimeIndex,X_coord,Y_coord,Z_coord
1,10.123,20.456,30.789
2,10.234,20.567,30.890
3,10.345,20.678,30.901
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
- **`distance_to_joint_centers.csv`**: Joint, X_velocity, Y_velocity, Z_velocity  
- **`gait_cycle_events.csv`**: EventNumber, EventIndex

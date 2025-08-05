# Gait Kinematics Function Output Documentation

## Overview

The `gait_kinematics.m` function is a foundational component of the Running Injury Clinic's gait analysis pipeline. It processes 3D marker data to calculate joint angles and angular velocities using anatomical coordinate systems and the Söderkvist method for segment tracking. This function transforms raw marker positions into biomechanically meaningful joint kinematics.

## Function Signature

```matlab
[angles, velocities, jc, R, djc] = gait_kinematics(joints, neutral, dynamic, hz, plots)
```

## Input Parameters

- **`joints`** (struct): Joint center locations from static calibration trial
- **`neutral`** (struct): Marker cluster positions from static calibration trial
- **`dynamic`** (struct): Marker cluster positions from dynamic gait trial
- **`hz`** (int): Data collection sampling frequency (Hz)
- **`plots`** (bool): Generate diagnostic visualizations (0=off, 1=on)

## Coordinate System Conventions

### Laboratory Coordinate System
- **X-axis**: Points to subject's right
- **Y-axis**: Points vertically upwards  
- **Z-axis**: Points opposite to walking direction

### Anatomical Segment Coordinate System
- **X-axis**: Anterior direction (Abduction/Adduction axis)
- **Y-axis**: Vertically upwards (Axial rotation axis)
- **Z-axis**: Points to subject's right (Flexion/Extension axis)

## Output Structure

### 1. `angles` - Joint Angles

**Type:** Struct  
**Purpose:** Calculated joint angles in degrees using Cardan angle sequence (XYZ rotation)

**Structure:**
```matlab
angles.L_ankle    % [N_frames × 3] - Left ankle joint angles
angles.R_ankle    % [N_frames × 3] - Right ankle joint angles
angles.L_knee     % [N_frames × 3] - Left knee joint angles
angles.R_knee     % [N_frames × 3] - Right knee joint angles
angles.L_hip      % [N_frames × 3] - Left hip joint angles
angles.R_hip      % [N_frames × 3] - Right hip joint angles
angles.L_foot     % [N_frames × 3] - Left foot segment angles
angles.R_foot     % [N_frames × 3] - Right foot segment angles
angles.pelvis     % [N_frames × 3] - Pelvis segment angles
```

**Dimensions:**
- **N_frames rows** = Number of time samples in dynamic trial
- **3 columns** = X, Y, Z rotational planes

**Units:** Degrees (°)

**Anatomical Interpretations:**

#### Joint Angles (Relative Motion)

##### Ankle
- **Ankle X (Frontal)**: Inversion(-) / Eversion(+)
- **Ankle Y (Transverse)**: Internal(-) / External(+) rotation
- **Ankle Z (Sagittal)**: Plantarflexion(-) / Dorsiflexion(+)

##### Knee
- **Knee X (Frontal)**: Adduction(-) / Abduction(+) 
- **Knee Y (Transverse)**: Internal(-) / External(+) rotation
- **Knee Z (Sagittal)**: Flexion(+) / Extension(-)

##### Hip
- **Hip X (Frontal)**: Adduction(-) / Abduction(+)
- **Hip Y (Transverse)**: Internal(-) / External(+) rotation  
- **Hip Z (Sagittal)**: Flexion(+) / Extension(-)

#### Segment Angles (Absolute Motion)

##### Foot
- **Foot X (Frontal)**: Foot eversion/inversion relative to lab
- **Foot Y (Transverse)**: Foot progression angle (toe-in/toe-out)
- **Foot Z (Sagittal)**: Foot pitch angle (heel/forefoot strike)

##### Pelvis
- **Pelvis X (Frontal)**: Pelvic obliquity (drop/hike)
- **Pelvis Y (Transverse)**: Pelvic rotation
- **Pelvis Z (Sagittal)**: Pelvic tilt

### 2. `velocities` - Joint Angular Velocities

**Type:** Struct  
**Purpose:** Joint angular velocities calculated from angle derivatives

**Structure:** Same field names as `angles`
```matlab
velocities.L_ankle    % [N_frames-1 × 3] - Left ankle angular velocities
velocities.R_ankle    % [N_frames-1 × 3] - Right ankle angular velocities
# ... (same fields as angles)
```

**Calculation Method:** First derivative of angles multiplied by sampling frequency
```matlab
velocities = diff(angles) * hz
```

**Dimensions:**
- **N_frames-1 rows** = One less frame due to differentiation
- **3 columns** = X, Y, Z rotational velocities

**Units:** Degrees per second (°/s)

### 3. `jc` - Joint Centers

**Type:** Struct  
**Purpose:** Calculated 3D joint center locations in laboratory coordinate system

**Structure:**
```matlab
jc.pelvis     % [1 × 3] - Pelvis center (average of 4 pelvic markers)
jc.L_hip      % [1 × 3] - Left hip joint center  
jc.R_hip      % [1 × 3] - Right hip joint center
jc.L_knee     % [1 × 3] - Left knee joint center
jc.R_knee     % [1 × 3] - Right knee joint center
jc.L_ankle    % [1 × 3] - Left ankle joint center
jc.R_ankle    % [1 × 3] - Right ankle joint center
```

**Calculation Methods:**
- **Pelvis**: Average of 4 pelvic markers
- **Hip**: 25% offset between left and right hip markers toward midline
- **Knee**: Midpoint between medial and lateral knee markers
- **Ankle**: Midpoint between medial and lateral ankle markers

**Units:** Millimeters (mm) in laboratory coordinate system

### 4. `R` - Rotation Matrices

**Type:** Struct  
**Purpose:** 4×4 homogeneous transformation matrices for each segment and joint

**Structure:**
```matlab
% Segment transformation matrices (anatomical to laboratory)
R.L_foot      % [4 × 4 × N_frames] - Left foot segment transformations
R.R_foot      % [4 × 4 × N_frames] - Right foot segment transformations  
R.L_shank     % [4 × 4 × N_frames] - Left shank segment transformations
R.R_shank     % [4 × 4 × N_frames] - Right shank segment transformations
R.L_thigh     % [4 × 4 × N_frames] - Left thigh segment transformations
R.R_thigh     % [4 × 4 × N_frames] - Right thigh segment transformations
R.pelvis      % [4 × 4 × N_frames] - Pelvis segment transformations

% Joint transformation matrices (relative motion between segments)
R.L_ankle     % [4 × 4 × N_frames] - Left ankle joint transformations
R.R_ankle     % [4 × 4 × N_frames] - Right ankle joint transformations
R.L_knee      % [4 × 4 × N_frames] - Left knee joint transformations  
R.R_knee      % [4 × 4 × N_frames] - Right knee joint transformations
R.L_hip       % [4 × 4 × N_frames] - Left hip joint transformations
R.R_hip       % [4 × 4 × N_frames] - Right hip joint transformations
```

**Matrix Format:**
```matlab
R = [R₃ₓ₃  T₃ₓ₁]
    [0₁ₓ₃  1   ]
```
Where:
- **R₃ₓ₃**: 3×3 rotation matrix
- **T₃ₓ₁**: 3×1 translation vector
- **Bottom row**: [0 0 0 1] for homogeneous coordinates

**Applications:**
- Forward kinematics calculations
- Coordinate system transformations
- Advanced biomechanical analysis

### 5. `djc` - Distance to Joint Centers

**Type:** Struct  
**Purpose:** Vector distances from segment centroids to joint centers in anatomical coordinates

**Structure:**
```matlab
djc.pelvis    % [3 × 1] - Pelvis centroid to pelvis center vector
djc.L_hip     % [3 × 1] - Left thigh centroid to left hip center vector
djc.R_hip     % [3 × 1] - Right thigh centroid to right hip center vector  
djc.L_knee    % [3 × 1] - Left shank centroid to left knee center vector
djc.R_knee    % [3 × 1] - Right shank centroid to right knee center vector
djc.L_ankle   % [3 × 1] - Left foot centroid to left ankle center vector
djc.R_ankle   % [3 × 1] - Right foot centroid to right ankle center vector
```

**Purpose:** 
- Enables joint center tracking during dynamic trials
- Used for calculating joint center positions from segment marker clusters
- Essential for accurate kinematic calculations

**Units:** Millimeters (mm) in anatomical coordinate system

## Technical Implementation Details

### Söderkvist Method
The function uses the Söderkvist and Wedin (1993) method for segment tracking:

1. **Anatomical Coordinate System Definition**: Establish segment coordinate systems from neutral trial
2. **Marker Registration**: Transform markers to anatomical coordinates  
3. **Dynamic Tracking**: Use Singular Value Decomposition (SVD) to find optimal rotation
4. **Transformation Matrices**: Calculate 4×4 homogeneous matrices for each frame

### Cardan Angle Calculation
Joint angles use the XYZ Cardan sequence:
```matlab
% Rotation matrix elements used for angle calculation:
% | CzCy-SzSySx  SzCy+CzSySx  -SyCx |
% | -SzCx        CzCx         Sx    |  
% | CzSy+SzCySx  SzSy-CzCySx  CyCx  |

x = atan2(R(2,3), sqrt(R(1,3)^2 + R(3,3)^2))  % Frontal plane
y = atan2(-R(1,3), R(3,3))                     % Transverse plane  
z = atan2(-R(2,1), R(2,2))                     % Sagittal plane
```

### Data Quality Considerations
- **Marker Visibility**: Assumes continuous marker tracking
- **Coordinate System Stability**: Requires proper marker placement
- **Gimbal Lock Avoidance**: atan2 function provides stability within physiological ranges

## Quality Control and Validation

### Expected Ranges (Typical Walking)
- **Ankle dorsiflexion**: 0-20°
- **Knee flexion**: 0-65°  
- **Hip flexion**: -15° to +30°
- **Foot progression**: ±15°

---

*This documentation corresponds to the gait_kinematics.m function in the Running Injury Clinic kinematic analysis pipeline.* 
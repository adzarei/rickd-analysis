# Gait Steps Function Output Documentation

## Overview

The `gait_steps.m` function is a core component of the Running Injury Clinic's gait analysis pipeline. It processes kinematic data to extract gait events, normalize step data, and calculate discrete biomechanical variables for clinical reporting.

## Function Signature

```matlab
[norm_ang, norm_vel, events, event, DISCRETE_VARIABLES, speedoutput, eventsflag, label] = gait_steps(neutral, dynamic, angles, velocities, hz, plots)
```

## Input Parameters

- **`neutral`** (struct): Marker positions from static calibration trial
- **`dynamic`** (struct): Marker positions from dynamic gait trial  
- **`angles`** (struct): Joint angles from `gait_kinematics` function
- **`velocities`** (struct): Joint velocities from `gait_kinematics` function
- **`hz`** (int): Data collection sampling frequency
- **`plots`** (bool): Generate diagnostic plots (0=off, 1=on)

## Output Structure

### 1. `norm_ang` - Time-Normalized Joint Direction Angles

**Type:** Struct  
**Purpose:** Joint angles normalized to 0-100% of stance phase (touchdown to toe-off)

**Structure:**
```matlab
norm_ang.L_ankle    % [101 × N_steps × 3] - Left ankle angles
norm_ang.L_knee     % [101 × N_steps × 3] - Left knee angles  
norm_ang.L_hip      % [101 × N_steps × 3] - Left hip angles
norm_ang.L_foot     % [101 × N_steps × 3] - Left foot angles
norm_ang.L_pelvis   % [101 × N_steps × 3] - Left pelvis angles
norm_ang.R_ankle    % [101 × N_steps × 3] - Right ankle angles
norm_ang.R_knee     % [101 × N_steps × 3] - Right knee angles
norm_ang.R_hip      % [101 × N_steps × 3] - Right hip angles
norm_ang.R_foot     % [101 × N_steps × 3] - Right foot angles
norm_ang.R_pelvis   % [101 × N_steps × 3] - Right pelvis angles
```

**Dimensions:**
- **101 rows** = 0-100% of stance phase (1% increments)
- **N_steps columns** = Number of detected gait steps  
- **3 planes** = X (frontal), Y (transverse), Z (sagittal) rotations

**Units:** Degrees (°)

**Note:** The angles are direction angles, which refers to the angles a vector makes with the x, y, and z axes. In this case joint.

### 2. `norm_vel` - Time-Normalized Joint Velocities

**Type:** Struct  
**Purpose:** Joint angular velocities normalized to 0-100% of stance phase

**Structure:** Same field names as `norm_ang` but contains angular velocities
```matlab
norm_vel.L_ankle    % [101 × N_steps × 3] - Left ankle velocities
norm_vel.L_knee     % [101 × N_steps × 3] - Left knee velocities
# ... (same fields as norm_ang)
```

**Units:** Degrees per second (°/s)

### 3. `events` - Gait Event Timing Matrix

**Type:** Matrix [N_steps × 4]  
**Purpose:** Frame indices of key gait events for stance phase analysis

**Structure:**
```matlab
events = [L_TD, L_TO, R_TD, R_TO]
```

| Column | Event | Description |
|--------|-------|-------------|
| 1 | L_TD | Left foot touchdown frame indices |
| 2 | L_TO | Left foot toe-off frame indices |
| 3 | R_TD | Right foot touchdown frame indices |
| 4 | R_TO | Right foot toe-off frame indices |

**Usage:** Each row represents one synchronized gait cycle. Convert to time using `frame_index / sampling_frequency`.

### 4. `event` - Extended Event Matrix with Midswing

**Type:** Matrix [N_steps × 8]  
**Purpose:** Expanded event matrix including midstance and heel whip timing for comprehensive gait analysis

**Structure:**
```matlab
event = [L_TD, L_Midstance, L_TO, L_heelwhip, R_TD, R_Midstance, R_TO, R_heelwhip]
```

| Column | Event | Description |
|--------|-------|-------------|
| 1 | L_TD | Left touchdown |
| 2 | L_Midstance | Left midstance (halfway between TD and TO) |
| 3 | L_TO | Left toe-off |
| 4 | L_heelwhip | Left maximum heel whip during swing |
| 5 | R_TD | Right touchdown |
| 6 | R_Midstance | Right midstance (halfway between TD and TO) |
| 7 | R_TO | Right toe-off |
| 8 | R_heelwhip | Right maximum heel whip during swing |

### 5. `DISCRETE_VARIABLES` - Biomechanical Parameters

**Type:** Matrix [77 × 3]  
**Purpose:** Calculated discrete biomechanical variables for clinical reporting and research

**Structure:**
- **Column 1:** Variable identifier (often unused/zero)
- **Column 2:** **Left side values**
- **Column 3:** **Right side values**

**Important Note:** Many rows are **empty placeholders** (remain as zeros) that were reserved for potential future variables. Only ~43% of the 77 variables are actually calculated.

**Variable Categories:**

#### Temporal-Spatial Parameters (Rows 2-6) - ALL POPULATED
- [POPULATED] Row 2: Step width (m)
- [POPULATED] Row 3: Stride rate (steps/min)
- [POPULATED] Row 4: Stride length (m)
- [POPULATED] Row 5: Swing time (s)
- [POPULATED] Row 6: Stance time (s)

#### Pelvis Kinematics (Rows 7-10) - PARTIALLY POPULATED
- [POPULATED] Row 7: Pelvis peak drop angle (°)
- [PLACEHOLDER] Row 8: Pelvis drop %stance *(placeholder)*
- [PLACEHOLDER] Row 9: Pelvis drop @HS *(placeholder)*
- [POPULATED] Row 10: Pelvis drop excursion (°)

#### Ankle Kinematics (Rows 11-22) - PARTIALLY POPULATED
- [POPULATED] Row 11: Ankle dorsiflexion peak angle (°)
- [PLACEHOLDER] Row 12: Ankle DF %stance *(placeholder)*
- [PLACEHOLDER] Row 13: Ankle DF @HS *(placeholder)*
- [PLACEHOLDER] Row 14: Ankle DF excursion *(placeholder)*
- [POPULATED] Row 15: Ankle eversion peak angle (°)
- [POPULATED] Row 16: Ankle eversion timing (%stance)
- [PLACEHOLDER] Row 17: Ankle eversion @HS *(placeholder)*
- [POPULATED] Row 18: Ankle eversion excursion (°)
- [POPULATED] Row 19: Ankle rotation peak angle (°)
- [PLACEHOLDER] Row 20: Ankle rotation %stance *(placeholder)*
- [PLACEHOLDER] Row 21: Ankle rotation @HS *(placeholder)*
- [POPULATED] Row 22: Ankle rotation excursion (°)

#### Knee Kinematics (Rows 23-38) - PARTIALLY POPULATED
- [POPULATED] Row 23: Knee flexion peak angle (°)
- [PLACEHOLDER] Row 24: Knee flexion %stance *(placeholder)*
- [PLACEHOLDER] Row 25: Knee flexion @HS *(placeholder)*
- [PLACEHOLDER] Row 26: Knee flexion excursion *(placeholder)*
- [POPULATED] Row 27: Knee adduction peak angle (°)
- [PLACEHOLDER] Row 28: Knee adduction %stance *(placeholder)*
- [PLACEHOLDER] Row 29: Knee adduction @HS *(placeholder)*
- [POPULATED] Row 30: Knee adduction excursion (°)
- [POPULATED] Row 31: Knee abduction peak angle (°)
- [PLACEHOLDER] Row 32: Knee abduction %stance *(placeholder)*
- [PLACEHOLDER] Row 33: Knee abduction @HS *(placeholder)*
- [POPULATED] Row 34: Knee abduction excursion (°)
- [POPULATED] Row 35: Knee rotation peak angle (°)
- [PLACEHOLDER] Row 36: Knee rotation %stance *(placeholder)*
- [PLACEHOLDER] Row 37: Knee rotation @HS *(placeholder)*
- [POPULATED] Row 38: Knee rotation excursion (°)

#### Hip Kinematics (Rows 39-50) - PARTIALLY POPULATED
- [POPULATED] Row 39: Hip extension peak angle (°)
- [PLACEHOLDER] Row 40: Hip extension %stance *(placeholder)*
- [PLACEHOLDER] Row 41: Hip extension @HS *(placeholder)*
- [PLACEHOLDER] Row 42: Hip extension excursion *(placeholder)*
- [POPULATED] Row 43: Hip adduction peak angle (°)
- [PLACEHOLDER] Row 44: Hip adduction %stance *(placeholder)*
- [PLACEHOLDER] Row 45: Hip adduction @HS *(placeholder)*
- [POPULATED] Row 46: Hip adduction excursion (°)
- [POPULATED] Row 47: Hip rotation peak angle (°)
- [PLACEHOLDER] Row 48: Hip rotation %stance *(placeholder)*
- [PLACEHOLDER] Row 49: Hip rotation @HS *(placeholder)*
- [POPULATED] Row 50: Hip rotation excursion (°)

#### Foot Kinematics (Rows 51-56) - PARTIALLY POPULATED
- [POPULATED] Row 51: Foot progression angle (°)
- [POPULATED] Row 52: Foot angle at heel strike (°)
- [PLACEHOLDER] Row 53: Foot angle @TO *(placeholder)*
- [PLACEHOLDER] Row 54: Med heel whip peak *(placeholder)*
- [PLACEHOLDER] Row 55: MHW %swing *(placeholder)*
- [POPULATED] Row 56: Medial heel whip excursion from toe-off (°)

#### Joint Velocities (Rows 57-76) - PARTIALLY POPULATED
- [PLACEHOLDER] Row 57: Ankle DF peak velocity *(placeholder)*
- [PLACEHOLDER] Row 58: Ankle DF velocity %stance *(placeholder)*
- [POPULATED] Row 59: Ankle eversion peak velocity (°/s)
- [PLACEHOLDER] Row 60: Ankle eversion velocity %stance *(placeholder)*
- [POPULATED] Row 61: Ankle rotation peak velocity (°/s)
- [PLACEHOLDER] Row 62: Ankle rotation velocity %stance *(placeholder)*
- [PLACEHOLDER] Row 63: Knee flexion peak velocity *(placeholder)*
- [PLACEHOLDER] Row 64: Knee flexion velocity %stance *(placeholder)*
- [POPULATED] Row 65: Knee abduction peak velocity (°/s)
- [PLACEHOLDER] Row 66: Knee abduction velocity %stance *(placeholder)*
- [POPULATED] Row 67: Knee adduction peak velocity (°/s)
- [PLACEHOLDER] Row 68: Knee adduction velocity %stance *(placeholder)*
- [POPULATED] Row 69: Hip abduction peak velocity (°/s)
- [PLACEHOLDER] Row 70: Hip abduction velocity %stance *(placeholder)*
- [POPULATED] Row 71: Knee rotation peak velocity (°/s)
- [POPULATED] Row 72: Hip rotation peak velocity (°/s)
- **Note**: Rows 73-74 are specialized timing variables (see section below)
- [POPULATED] Row 75: Hip adduction peak velocity (°/s)
- [POPULATED] Row 76: Pelvic drop peak velocity (°/s)

#### Specialized Variables (Rows 73-74, 77) - ALL POPULATED
- [POPULATED] Row 73: Pronation onset timing (%stance)
- [POPULATED] Row 74: Supination timing (%stance)
- [POPULATED] Row 77: Vertical oscillation (mm)

#### Summary Statistics
- **Populated Variables**: 33 out of 77 (~43%)
- **Empty Placeholders**: 44 out of 77 (~57%)

**Note:** Empty placeholders were intentionally reserved for potential future variables without requiring changes to the data structure. The comment in the source code states: *"Note some variables will remain empty. These were used as placeholders in case those variables were determined to be of interest at a later date."*

### 6. `speedoutput` - Gait Speed

**Type:** Float  
**Purpose:** Calculated walking/running speed

**Value:** Speed in meters per second (m/s) based on heel marker displacement  
**Method:** Calculated from anterior-posterior position of the lowest heel marker

### 7. `eventsflag` - Event Detection Method Quality

**Type:** Matrix [N_steps × 4]  
**Purpose:** Indicates reliability of event detection for each gait event

**Structure:** Same dimensions as `events` matrix

**Values:**
- **0** = Default foot-forward/foot-back detection used (less accurate)
- **1** = PCA-based event detection used (more accurate)

**Interpretation:** Higher proportion of 1s indicates more reliable gait event detection.

### 8. `label` - Gait Classification

**Type:** String  
**Purpose:** Automatic classification of locomotion type

**Possible Values:**
- **'walk'** = Walking gait pattern detected
- **'run'** = Running gait pattern detected

**Classification Method:** Uses trained Linear Discriminant Analysis (LDA) classifier based on:
- Gait speed (m/s)
- Stride rate (steps/min)

## Data Quality and Processing Notes

### Outlier Removal
- The function automatically removes gait cycles that are >3 standard deviations from the mean
- Uses histogram-based filtering for bimodal data distributions
- Ensures only high-quality, consistent gait cycles are retained

### Event Detection Hierarchy
1. **Primary:** PCA-based touchdown/toe-off detection (most accurate)
2. **Fallback:** Foot-forward/foot-back detection (more robust)
3. **Quality flagged** in `eventsflag` output

### Normalization Method
- Stance phase normalized to 101 points (0-100%)
- Uses piecewise cubic Hermite interpolation (`pchip`)
- Preserves biomechanical curve characteristics

---

*This documentation corresponds to the gait_steps.m function in the Running Injury Clinic kinematic analysis pipeline.* 
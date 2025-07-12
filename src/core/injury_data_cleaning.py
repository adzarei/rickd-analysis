"""
Injury data cleaning module for the RICKD analysis project.

This module provides comprehensive data cleaning functions for injury-related data,
implementing all the requirements specified in the comments.
"""

import pandas as pd
import numpy as np
from core.data_quality import map_injury_codes
from core.constants import RICKD_MAP_INJURY_CODE, RICKD_MAP_INJURY_DESC, RICKD_MAP_SELF_INJURY_CODE


def clean_injury_data(df):
    """
    Comprehensive data cleaning function for injury-related data.
    
    This function implements all the cleaning requirements:
    - Convert injury severity to numeric scale 1-4
    - Clean and consolidate joint and side information
    - Apply injury code mappings and additional injury consolidations
    - Clean duration, activities, level, and other variables
    - Extract healthy/injured classification
    
    Args:
        df: DataFrame containing injury data
        
    Returns:
        DataFrame with cleaned injury data including new columns:
        - injury_severity: Numeric severity scale 1-4
        - InjJoint_clean, InjJoint2_clean: Standardized joint categories
        - InjSide_clean, InjSide2_clean: Standardized side categories
        - injury_code, injury2_code: Mapped injury codes
        - injury_desc, injury2_desc: Injury descriptions
        - InjDuration_clean: Cleaned duration values
        - Activities_clean: Standardized activity categories
        - Level_clean: Standardized level categories
        - YrsRunning_clean: Cleaned years running
        - NumRaces_clean: Cleaned number of races
        - is_injured: Binary injury classification (0=healthy, 1=injured)
    """
    df_clean = df.copy()
    
    # 1. Convert injury severity (InjDefn) to numeric scale 1-4
    severity_mapping = {
        'no injury': 1,
        'training volume/intensity affected': 2,
        'continuing to train in pain': 3,
        '2 workouts missed in a row': 4
    }
    df_clean['injury_severity'] = df_clean['InjDefn'].map(severity_mapping)
    
    # 2. Clean and consolidate InjJoint and InjJoint2
    # Standardize joint categories
    joint_mapping = {
        'ankle': 'ankle',
        'foot': 'foot',
        'hip': 'hip',
        'hip/pelvis': 'hip_pelvis',
        'knee': 'knee',
        'lumbar spine': 'lumbar_spine',
        'thigh': 'thigh',
        'no injury': 'no_injury',
        'no injury,no injury': 'no_injury',
        'no injury, no injury': 'no_injury',
        'no injury,no injury,no injury': 'no_injury'
    }
    
    df_clean['InjJoint_clean'] = df_clean['InjJoint'].str.lower().map(joint_mapping)
    df_clean['InjJoint2_clean'] = df_clean['InjJoint2'].str.lower().map(joint_mapping)
    
    # Only populate for actual injuries
    df_clean.loc[df_clean['injury_severity'] == 1, ['InjJoint_clean', 'InjJoint2_clean']] = 'no_injury'
    
    # 3. Clean and consolidate InjSide and InjSide2
    side_mapping = {
        'left': 'left',
        'right': 'right',
        'bilateral': 'both',
        'both': 'both'
    }
    
    df_clean['InjSide_clean'] = df_clean['InjSide'].str.lower().map(side_mapping)
    df_clean['InjSide2_clean'] = df_clean['InjSide2'].str.lower().map(side_mapping)
    
    # Only populate for actual injuries
    df_clean.loc[df_clean['injury_severity'] == 1, ['InjSide_clean', 'InjSide2_clean']] = None
    
    # 4. Clean SpecInjury and SpecInjury2 - convert to lowercase and apply additional mappings
    # Additional injury mappings as specified in comments
    additional_injury_mapping = {
        'oa': 'osteoarthritis',
        'hip oa': 'osteoarthritis',
        'knee oa': 'osteoarthritis',
        'osteroarthritis': 'osteoarthritis',
        'itbs': 'itb syndrome',
        'pfps': 'patellofemoral pain syndrome'
    }
    
    # Apply additional mappings first
    for old_val, new_val in additional_injury_mapping.items():
        # Handle NaN values properly
        mask = df_clean['SpecInjury'].notna()
        df_clean.loc[mask, 'SpecInjury'] = df_clean.loc[mask, 'SpecInjury'].str.replace(old_val, new_val, case=False)
        
        mask2 = df_clean['SpecInjury2'].notna()
        df_clean.loc[mask2, 'SpecInjury2'] = df_clean.loc[mask2, 'SpecInjury2'].str.replace(old_val, new_val, case=False)
    
    # Convert to lowercase
    df_clean['SpecInjury'] = df_clean['SpecInjury'].str.lower()
    df_clean['SpecInjury2'] = df_clean['SpecInjury2'].str.lower()
    
    # 5. Map injury codes using existing mapping tables
    df_clean = map_injury_codes(df_clean, RICKD_MAP_INJURY_CODE, "SpecInjury", "injury_code")
    df_clean = map_injury_codes(df_clean, RICKD_MAP_INJURY_CODE, "SpecInjury2", "injury2_code")
    df_clean = map_injury_codes(df_clean, RICKD_MAP_INJURY_DESC, "injury_code", "injury_desc")
    df_clean = map_injury_codes(df_clean, RICKD_MAP_INJURY_DESC, "injury2_code", "injury2_desc")
    
    # 6. Clean InjDuration - convert to reasonable values
    # Remove extreme outliers and convert to numeric
    df_clean['InjDuration_clean'] = pd.to_numeric(df_clean['InjDuration'], errors='coerce')
    
    # Remove unreasonable values (e.g., > 10 years)
    df_clean.loc[df_clean['InjDuration_clean'] > 3650, 'InjDuration_clean'] = np.nan
    
    # 7. Clean Activities - consolidate into established categories
    activity_mapping = {
        'running': 'running',
        'swimming': 'swimming',
        'cycling': 'cycling',
        'yoga': 'yoga',
        'pilates': 'pilates',
        'strength training': 'strength_training',
        'walking': 'walking',
        'hiking': 'hiking',
        'triathlon': 'triathlon',
        'track': 'track',
        'power walking': 'power_walking',
        'horseback riding': 'horseback_riding',
        'sprint triathlon': 'triathlon'
    }
    
    df_clean['Activities_clean'] = df_clean['Activities'].str.lower().map(activity_mapping)
    
    # 8. Clean Level - consolidate into recreational and competitive
    level_mapping = {
        'recreational': 'recreational',
        'competitive': 'competitive'
    }
    
    df_clean['Level_clean'] = df_clean['Level'].str.lower().map(level_mapping)
    
    # 9. Clean YrsRunning - convert to reasonable values
    df_clean['YrsRunning_clean'] = pd.to_numeric(df_clean['YrsRunning'], errors='coerce')
    
    # Remove unreasonable values (e.g., > 50 years)
    df_clean.loc[df_clean['YrsRunning_clean'] > 50, 'YrsRunning_clean'] = np.nan
    
    # 10. Clean NumRaces - convert to reasonable values
    df_clean['NumRaces_clean'] = pd.to_numeric(df_clean['NumRaces'], errors='coerce')
    
    # Remove unreasonable values (e.g., > 1000 races)
    df_clean.loc[df_clean['NumRaces_clean'] > 1000, 'NumRaces_clean'] = np.nan
    
    # 11. Extract healthy/injured variable
    # A subject is considered UNINJURED if:
    # - InjDefn = 'No injury' AND
    # - InjJoint = 'No injury' or empty AND
    # - SpecInjury is empty
    df_clean['is_injured'] = ~(
        (df_clean['InjDefn'] == 'no injury') &
        (df_clean['InjJoint'].isin(['no injury', 'no injury,no injury', 'no injury, no injury', 'no injury,no injury,no injury']) | df_clean['InjJoint'].isna()) &
        (df_clean['SpecInjury'].isna())
    )
    
    # Convert to numeric for easier analysis
    df_clean['is_injured'] = df_clean['is_injured'].astype(int)
    
    return df_clean


def get_cleaning_summary(df_clean):
    """
    Generate a summary of the cleaning results.
    
    Args:
        df_clean: DataFrame with cleaned injury data
        
    Returns:
        dict: Summary statistics of the cleaning process
    """
    summary = {
        'total_records': len(df_clean),
        'injury_severity_distribution': df_clean['injury_severity'].value_counts().sort_index().to_dict(),
        'injured_vs_uninjured': df_clean['is_injured'].value_counts().to_dict(),
        'top_injury_codes': df_clean['injury_code'].value_counts().head(10).to_dict(),
        'activity_levels': df_clean['Level_clean'].value_counts().to_dict(),
        'joint_distribution': df_clean['InjJoint_clean'].value_counts().to_dict(),
        'side_distribution': df_clean['InjSide_clean'].value_counts().to_dict()
    }
    
    return summary


def print_cleaning_summary(df_clean):
    """
    Print a formatted summary of the cleaning results.
    
    Args:
        df_clean: DataFrame with cleaned injury data
    """
    print("=== Injury Data Cleaning Summary ===")
    print(f"Total records: {len(df_clean)}")
    
    print(f"\nInjury severity distribution:")
    severity_counts = df_clean['injury_severity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        severity_name = {
            1: 'No injury',
            2: 'Training volume/intensity affected', 
            3: 'Continuing to train in pain',
            4: '2 workouts missed in a row'
        }.get(severity, f'Unknown ({severity})')
        print(f"  {severity_name}: {count}")

    print(f"\nInjured vs Uninjured:")
    injury_counts = df_clean['is_injured'].value_counts()
    print(f"  Injured: {injury_counts.get(1, 0)}")
    print(f"  Uninjured: {injury_counts.get(0, 0)}")

    print(f"\nTop injury codes:")
    top_injuries = df_clean['injury_code'].value_counts().head(10)
    for injury, count in top_injuries.items():
        print(f"  {injury}: {count}")

    print(f"\nActivity levels:")
    level_counts = df_clean['Level_clean'].value_counts()
    for level, count in level_counts.items():
        print(f"  {level}: {count}")
    
    print(f"\nJoint distribution:")
    joint_counts = df_clean['InjJoint_clean'].value_counts()
    for joint, count in joint_counts.items():
        print(f"  {joint}: {count}")
    
    print(f"\nSide distribution:")
    side_counts = df_clean['InjSide_clean'].value_counts()
    for side, count in side_counts.items():
        print(f"  {side}: {count}") 
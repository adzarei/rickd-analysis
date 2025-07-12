#!/usr/bin/env python3
"""
Simple test script to verify the injury data cleaning function works correctly.
"""

import pandas as pd
import numpy as np
from core.injury_data_cleaning import clean_injury_data, print_cleaning_summary

# Create a small test dataset
test_data = pd.DataFrame({
    'sub_id': [1001, 1002, 1003, 1004, 1005],
    'InjDefn': ['no injury', 'training volume/intensity affected', 'continuing to train in pain', '2 workouts missed in a row', 'no injury'],
    'InjJoint': ['no injury', 'knee', 'ankle', 'hip/pelvis', 'no injury,no injury'],
    'InjSide': ['right', 'left', 'bilateral', 'right', None],
    'SpecInjury': [None, 'itb syndrome', 'pfps', 'osteoarthritis', None],
    'SpecInjury2': [None, None, 'pain', None, None],
    'InjDuration': [None, 30, 60, 90, None],
    'Activities': ['running', 'swimming', 'yoga', 'cycling', 'walking'],
    'Level': ['recreational', 'competitive', 'recreational', 'competitive', 'recreational'],
    'YrsRunning': [5, 10, 2, 15, 1],
    'NumRaces': [10, 50, 5, 100, 2]
})

print("Original test data:")
print(test_data)
print("\n" + "="*50)

# Apply cleaning
cleaned_test = clean_injury_data(test_data)

print("Cleaned test data:")
print(cleaned_test[['sub_id', 'injury_severity', 'is_injured', 'injury_code', 'InjJoint_clean', 'InjSide_clean']])

print("\n" + "="*50)
print_cleaning_summary(cleaned_test)

print("\nTest completed successfully!") 
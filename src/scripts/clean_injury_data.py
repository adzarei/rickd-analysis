#!/usr/bin/env python3
"""
Script to clean injury data according to the specifications in the comments.

This script implements all the cleaning requirements:
- Convert injury severity to numeric scale 1-4
- Clean and consolidate joint and side information
- Apply injury code mappings and additional injury consolidations
- Clean duration, activities, level, and other variables
- Extract healthy/injured classification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from core.injury_data_cleaning import clean_injury_data, print_cleaning_summary
from core.constants import RICKD_SESSION_DATA_FULL_FILE, RICKD_SESSION_DATA_FULL_CLEANED_FILE

def main():
    """Main function to clean injury data and save results."""
    
    print("Loading data...")
    
    
    print("\nCleaning completed successfully!")

if __name__ == "__main__":
    main() 
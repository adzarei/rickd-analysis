#!/usr/bin/env python3
"""
Test script to compare MATLAB and Python implementations
of the gait kinematics processing pipeline.

This script loads the MATLAB outputs saved by processing_code_example_with_outputs.m
and compares them with the Python implementation results.
"""

import numpy as np
import json
import os
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Add the src directory to the path to import our modules
import sys
sys.path.append('src')

from core.kinematics import gait_kinematics, processing_pipeline


def load_matlab_results(matlab_output_dir: str, subject_name: str) -> Dict[str, Any]:
    """Load MATLAB results from .mat files"""
    
    results = {}
    
    # Load main results
    main_file = os.path.join(matlab_output_dir, f"{subject_name}_matlab_results.mat")
    if os.path.exists(main_file):
        results['main'] = loadmat(main_file, struct_as_record=False, squeeze_me=True)
    
    # Load inputs
    inputs_file = os.path.join(matlab_output_dir, f"{subject_name}_inputs.mat")
    if os.path.exists(inputs_file):
        results['inputs'] = loadmat(inputs_file, struct_as_record=False, squeeze_me=True)
    
    # Load walking results if available
    walking_file = os.path.join(matlab_output_dir, f"{subject_name}_walking_results.mat")
    if os.path.exists(walking_file):
        results['walking'] = loadmat(walking_file, struct_as_record=False, squeeze_me=True)
    
    # Load running results if available
    running_file = os.path.join(matlab_output_dir, f"{subject_name}_running_results.mat")
    if os.path.exists(running_file):
        results['running'] = loadmat(running_file, struct_as_record=False, squeeze_me=True)
    
    return results


def matlab_struct_to_dict(struct_obj) -> Dict:
    """Convert MATLAB struct to Python dictionary"""
    result = {}
    
    for field_name in struct_obj._fieldnames:
        field_value = getattr(struct_obj, field_name)
        
        if hasattr(field_value, '_fieldnames'):
            # Nested struct
            result[field_name] = matlab_struct_to_dict(field_value)
        else:
            # Regular value
            result[field_name] = field_value
    
    return result


def compare_arrays(matlab_array: np.ndarray, python_array: np.ndarray, 
                  name: str, tolerance: float = 1e-10) -> bool:
    """Compare two arrays and report differences"""
    
    if matlab_array.shape != python_array.shape:
        print(f"❌ {name}: Shape mismatch - MATLAB: {matlab_array.shape}, Python: {python_array.shape}")
        return False
    
    # Calculate differences
    diff = np.abs(matlab_array - python_array)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    if max_diff < tolerance:
        print(f"✅ {name}: Arrays match (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"❌ {name}: Arrays differ (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        
        # Show some statistics
        print(f"   MATLAB range: [{np.min(matlab_array):.6f}, {np.max(matlab_array):.6f}]")
        print(f"   Python range: [{np.min(python_array):.6f}, {np.max(python_array):.6f}]")
        
        return False


def test_gait_kinematics_outputs(matlab_results: Dict, python_results: Dict, 
                                gait_type: str = 'walking') -> Dict[str, bool]:
    """Test gait_kinematics outputs between MATLAB and Python"""
    
    print(f"\n🔍 Testing gait_kinematics outputs for {gait_type}...")
    
    test_results = {}
    
    # Get MATLAB results
    if gait_type in matlab_results and 'gait_kinematics' in matlab_results[gait_type]:
        matlab_gk = matlab_results[gait_type]['gait_kinematics']
    else:
        print(f"❌ No MATLAB {gait_type} gait_kinematics results found")
        return test_results
    
    # Get Python results
    if gait_type in python_results:
        python_gk = python_results[gait_type]
    else:
        print(f"❌ No Python {gait_type} results found")
        return test_results
    
    # Test angles
    if hasattr(matlab_gk, 'angles') and 'angles' in python_gk:
        matlab_angles = matlab_struct_to_dict(matlab_gk.angles)
        python_angles = python_gk['angles']
        
        for joint in ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip']:
            if joint in matlab_angles and joint in python_angles:
                test_results[f'angles_{joint}'] = compare_arrays(
                    matlab_angles[joint], python_angles[joint], 
                    f"{gait_type} angles {joint}"
                )
    
    # Test velocities
    if hasattr(matlab_gk, 'velocities') and 'velocities' in python_gk:
        matlab_velocities = matlab_struct_to_dict(matlab_gk.velocities)
        python_velocities = python_gk['velocities']
        
        for joint in ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip']:
            if joint in matlab_velocities and joint in python_velocities:
                test_results[f'velocities_{joint}'] = compare_arrays(
                    matlab_velocities[joint], python_velocities[joint], 
                    f"{gait_type} velocities {joint}"
                )
    
    # Test joint centers
    if hasattr(matlab_gk, 'joint_centers') and 'joint_centers' in python_gk:
        matlab_jc = matlab_struct_to_dict(matlab_gk.joint_centers)
        python_jc = python_gk['joint_centers']
        
        for joint in ['pelvis', 'L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']:
            if joint in matlab_jc and joint in python_jc:
                test_results[f'jc_{joint}'] = compare_arrays(
                    matlab_jc[joint], python_jc[joint], 
                    f"{gait_type} joint center {joint}"
                )
    
    return test_results


def create_comparison_plots(matlab_results: Dict, python_results: Dict, 
                           gait_type: str, output_dir: str):
    """Create comparison plots between MATLAB and Python results"""
    
    print(f"\n📊 Creating comparison plots for {gait_type}...")
    
    # Create output directory
    plot_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get data
    if gait_type in matlab_results and 'gait_kinematics' in matlab_results[gait_type]:
        matlab_gk = matlab_results[gait_type]['gait_kinematics']
        matlab_angles = matlab_struct_to_dict(matlab_gk.angles)
    else:
        print(f"No MATLAB {gait_type} data for plotting")
        return
    
    if gait_type in python_results and 'angles' in python_results[gait_type]:
        python_angles = python_results[gait_type]['angles']
    else:
        print(f"No Python {gait_type} data for plotting")
        return
    
    # Plot joint angles
    joints = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip']
    planes = ['x', 'y', 'z']
    
    for joint in joints:
        if joint in matlab_angles and joint in python_angles:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'{gait_type.title()} - {joint} Joint Angles: MATLAB vs Python')
            
            for i, plane in enumerate(planes):
                axes[i].plot(matlab_angles[joint][:, i], 'b-', label='MATLAB', linewidth=2)
                axes[i].plot(python_angles[joint][:, i], 'r--', label='Python', linewidth=2)
                axes[i].set_title(f'{plane.upper()}-axis')
                axes[i].set_ylabel('Angle (degrees)')
                axes[i].legend()
                axes[i].grid(True)
            
            axes[-1].set_xlabel('Time (samples)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{gait_type}_{joint}_angles.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main testing function"""
    
    print("🧪 Testing MATLAB vs Python Gait Kinematics Implementation")
    print("=" * 60)
    
    # Configuration
    matlab_output_dir = "/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/matlab_outputs"
    json_data_dir = "/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/source_data"
    
    # Find the first JSON file
    json_files = []
    for root, dirs, files in os.walk(json_data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print("❌ No JSON files found in data directory")
        return
    
    # Use the first JSON file
    json_file = json_files[0]
    subject_name = Path(json_file).stem
    
    print(f"📁 Testing with subject: {subject_name}")
    print(f"📄 JSON file: {json_file}")
    
    # Load MATLAB results
    print("\n📥 Loading MATLAB results...")
    try:
        matlab_results = load_matlab_results(matlab_output_dir, subject_name)
        if 'main' in matlab_results:
            matlab_data = matlab_struct_to_dict(matlab_results['main']['results'])
            print("✅ MATLAB results loaded successfully")
        else:
            print("❌ Could not load MATLAB main results")
            return
    except Exception as e:
        print(f"❌ Error loading MATLAB results: {e}")
        return
    
    # Run Python implementation
    print("\n🐍 Running Python implementation...")
    try:
        python_results = processing_pipeline(json_file, plots=False)
        print("✅ Python implementation completed successfully")
    except Exception as e:
        print(f"❌ Error running Python implementation: {e}")
        return
    
    # Compare results
    print("\n🔍 Comparing Results...")
    print("-" * 40)
    
    all_tests_passed = True
    
    # Test walking data if available
    if 'walking' in matlab_data and 'walking' in python_results:
        walking_tests = test_gait_kinematics_outputs(matlab_data, python_results, 'walking')
        if not all(walking_tests.values()):
            all_tests_passed = False
        
        # Create comparison plots
        try:
            create_comparison_plots(matlab_data, python_results, 'walking', matlab_output_dir)
        except Exception as e:
            print(f"⚠️  Could not create walking plots: {e}")
    
    # Test running data if available
    if 'running' in matlab_data and 'running' in python_results:
        running_tests = test_gait_kinematics_outputs(matlab_data, python_results, 'running')
        if not all(running_tests.values()):
            all_tests_passed = False
        
        # Create comparison plots
        try:
            create_comparison_plots(matlab_data, python_results, 'running', matlab_output_dir)
        except Exception as e:
            print(f"⚠️  Could not create running plots: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Python implementation matches MATLAB.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review the differences above.")
    
    print(f"\n📊 Comparison plots saved in: {os.path.join(matlab_output_dir, 'comparison_plots')}")


if __name__ == "__main__":
    main() 
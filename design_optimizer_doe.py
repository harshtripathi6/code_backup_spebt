import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

# --- Configuration ---
# Create a folder to hold all the .tensor files and .hdf5 results
WORK_DIR = "sensitivity_study_runs"
os.makedirs(WORK_DIR, exist_ok=True)

# --- 1. Define Search Space ---
pinhole_diameters = np.linspace(1.0, 5.0, 5) # 1mm to 5mm
outward_displacements = np.linspace(0, 40, 5) # 0mm to 40mm

print(f"Starting Design Sensitivity Study: {len(pinhole_diameters) * len(outward_displacements)} configurations.")
print(f"Data will be saved to: {os.path.abspath(WORK_DIR)}")

# --- 2. The Execution Loop ---
for d in pinhole_diameters:
    for disp in outward_displacements:
        # Create a unique ID for this specific configuration
        # e.g., "d3p0_disp10p0"
        run_id = f"d{d:.1f}_disp{disp:.1f}".replace('.', 'p')
        
        print(f"\n>>> Evaluating: Diameter={d}mm, Displacement={disp}mm [ID: {run_id}]")

        # STEP A: Module 1 - Configuration Generation
        # We pass output_dir so the script knows where to save the .tensor file
        cmd_mod1 = [
            "python", "generate_mph_scanner_circularfov.py",
            "--diameter", str(d),
            "--displacement", str(disp),
            "--run_id", run_id,
            "--out_dir", WORK_DIR 
        ]
        
        try:
            subprocess.run(cmd_mod1, check=True)
        except subprocess.CalledProcessError:
            print(f"!!! Error generating geometry for {run_id}. Skipping.")
            continue

        # STEP B: Module 2 - PPDF Computation
        # We target layout 0. We pass the same run_id and out_dir
        cmd_mod2 = [
            "python", "arg_ppdf_calculation.py", "0", 
            "--run_id", run_id,
            "--work_dir", WORK_DIR
        ]
        
        try:
            subprocess.run(cmd_mod2, check=True)
        except subprocess.CalledProcessError:
            print(f"!!! Error calculating PPDF for {run_id}. Skipping.")
            continue

print("\nStudy Complete.")
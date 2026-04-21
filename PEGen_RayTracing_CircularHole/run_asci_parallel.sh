#!/bin/bash

#SBATCH --job-name=asci_3d_gen         # Job name for identification
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=02:00:00                # Walltime limit (HH:MM:SS)
#SBATCH --nodes=1                      # Run all tasks on a single node
#SBATCH --ntasks=1                     # Request 1 task (our python script)
#SBATCH --cpus-per-task=4              # Adjusted: 4 cores should be plenty for h5py/numpy
#SBATCH --mem=8G                       # Adjusted: 8 GB of memory to safely hold the 3D histograms
#SBATCH --array=0-59                   # Creates a job array for 60 rotations, indexed 0-59
#SBATCH --mail-user=htripath@buffalo.edu
#SB# %A is the main job ID, %a is the array task ID
#SBATCH --output=slurm_logs/out/asci_%A_%a.out
#SBATCH --error=slurm_logs/err/asci_%A_%a.err

# --- Output/Error Logging ---
# Create a directory for logs if it doesn't exist
mkdir -p slurm_logs/out slurm_logs/err


# --- Environment Setup ---
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start Time: $(date)"
echo "=========================================================="

# Activate your Python environment
source /user/htripath/venv/bin/activate
ml intel gcccore/11.2.0 python/3.9.6-bare hdf5/1.14.1

# --- Execute the Python Script ---
# Slurm runs this command for each task ID in the array (0, 1, 2, ..., 59)
echo "Executing Python script for rotation index $SLURM_ARRAY_TASK_ID..."

python generate_3d_asci.py $SLURM_ARRAY_TASK_ID

echo "=========================================================="
echo "End Time: $(date)"
echo "Job finished with exit code $?"
echo "=========================================================="
#!/bin/bash

#SBATCH --job-name=beam_gen        # Job name for identification
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=02:00:00             # Walltime limit (HH:MM:SS), e.g., 2 hours
#SBATCH --nodes=1                   # Run all tasks on a single node
#SBATCH --ntasks=1                  # Request 1 task (our python script)
#SBATCH --cpus-per-task=1           # Request 1 CPU core per task
#SBATCH --mem=5G                    # Request 3 GB of memory per task
#SBATCH --array=0-14            

# --- Output/Error Logging ---
# Create a directory for logs
mkdir -p slurm_logs
# %A is the main job ID, %a is the array task ID
#SBATCH --output=slurm_logs/beam_proc_%A_%a.out
#SBATCH --error=slurm_logs/beam_proc_%A_%a.err

# --- Environment Setup ---
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start Time: $(date)"
echo "=========================================================="

# Activate your Python environment (e.g., Conda)
source /user/htripath/venv/bin/activate
ml intel gcccore/11.2.0 python/3.9.6-bare hdf5/1.14.1
pip install --user mpi4py
CC=mpicc HDF5_MPI="ON" HDF5_DIR="/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/avx512/MPI/intel/2022.0.1/impi/2021.5.0/hdf5/1.14.1" pip install --user --no-binary=h5py h5py  

# --- Execute the Python Script ---
# Slurm will run this command for each task in the array,
# substituting $SLURM_ARRAY_TASK_ID with the current index (0, 1, ... 5).
# Step 1: Generate Beam Masks
echo "--- [START] Step 1: Generating Beam Masks for layout $SLURM_ARRAY_TASK_ID ---"
python arg_extract_beam_masks.py $SLURM_ARRAY_TASK_ID
echo "--- [ END ] Step 1: Finished Beam Masks for layout $SLURM_ARRAY_TASK_ID ---"
echo ""

# Step 2: Extract Beam Properties (depends on Step 1)
echo "--- [START] Step 2: Extracting Beam Properties for layout $SLURM_ARRAY_TASK_ID ---"
python arg_extract_beam_properties.py $SLURM_ARRAY_TASK_ID
echo "--- [ END ] Step 2: Finished Beam Properties for layout $SLURM_ARRAY_TASK_ID ---"
echo ""

# Step 3: Generate ASCI Histogram (depends on Steps 1 & 2)
echo "--- [START] Step 3: Generating ASCI Histogram for layout $SLURM_ARRAY_TASK_ID ---"
python arg_analyze_extracted_properties.py $SLURM_ARRAY_TASK_ID
echo "--- [ END ] Step 3: Finished ASCI Histogram for layout $SLURM_ARRAY_TASK_ID ---"
echo ""


echo "=========================================================="
echo "End Time: $(date)"
echo "=========================================================="
#!/bin/bash
#SBATCH --job-name=pe_sysmat
#SBATCH --partition=eai-test
#SBATCH --qos=eai-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H100   # Change to H100 or A100 depending on availability
#SBATCH --output=pe_%j.out
#SBATCH --error=pe_%j.err

echo "Starting Job: $SLURM_JOB_ID on node $SLURMD_NODENAME"
echo "Allocated GPU Hardware:"
nvidia-smi -L
echo "----------------------------------------------------------"
# 1. Load Cluster Environment (Explicit versions for reproducibility)
module purge
module load ccrsoft/2024.04
module load gcc cuda

# 2. Navigate to the correct directory
# Adjust this path if you submit from outside the project folder
cd PEGen_RayTracing_CircularHole || { echo "Directory not found!"; exit 1; }

# 3. Pre-run Validation (Crucial!)
# Ensure all parameter files exist before wasting GPU time compiling
for file in Params_Physics.dat Params_Image.dat Params_Collimator.dat Params_Detector.dat; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing $file! Did you run the python generators?"
        exit 1
    fi
done

# 4. Compile components (Using Script 1's robust static linking)
echo "Compiling GPU and CPU codes..."

# Optional: Add compute capability flags for maximum performance. 
# V100 = -arch=sm_70 | A100 = -arch=sm_80 | H100 = -arch=sm_90
nvcc -O3 -arch=sm_90 -c PESysMatGen.cu -o PESysMatGen_cu.o
g++ -O3 -c PEGen_CircularHole.cpp -o PEGen_cpp.o

# Link everything together safely
nvcc -o PESysMatGen PEGen_cpp.o PESysMatGen_cu.o \
    -lcudart_static -ldl -lrt -lpthread \
    -Xcompiler "-static-libgcc -static-libstdc++"

# 5. Run the Simulation
if [ -f PESysMatGen ]; then
    echo "Compilation successful. Launching GPUPTS simulation..."
    
    # Measure exactly how long the GPU execution takes
    time srun ./PESysMatGen -cuda 0
    
    echo "Simulation complete! Check the directory for output matrix files."
else
    echo "ERROR: Compilation failed. Check the error log."
    exit 1
fi
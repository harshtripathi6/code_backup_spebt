#!/bin/bash
#SBATCH --job-name=pe_sysmat_v3
#SBATCH --partition=eai-test
#SBATCH --qos=eai-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H100
#SBATCH --output=pe_%j.out
#SBATCH --error=pe_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU:"
nvidia-smi -L

module purge
module load ccrsoft
module load gcc cuda

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${ROOT_DIR}/PEGen_RayTracing_CircularHole"

cd "${WORK_DIR}"

for file in \
    Params_Physics.dat \
    Params_Image.dat \
    Params_Collimator.dat \
    Params_Detector.dat
do
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: ${file} is missing."
        echo "Run the four parameter-generation scripts before submitting this job."
        exit 1
    fi
done

echo "Compiling PESysMatGen_v3.cu and PEGen_CircularHole.cpp..."

rm -f PESysMatGen_v3.o PEGen_CircularHole.o PESysMatGen_v3

nvcc -O3 -arch=sm_90 \
    -c PESysMatGen_v3.cu \
    -o PESysMatGen_v3.o

g++ -O3 \
    -c PEGen_CircularHole.cpp \
    -o PEGen_CircularHole.o

nvcc -O3 -arch=sm_90 \
    -o PESysMatGen_v3 \
    PEGen_CircularHole.o \
    PESysMatGen_v3.o \
    -lcudart_static -ldl -lrt -lpthread \
    -Xcompiler "-static-libgcc -static-libstdc++"

echo "Compilation completed."
echo "Starting system-matrix generation..."

time srun ./PESysMatGen_v3 -cuda 0

echo "System-matrix generation completed."
echo "Generated files:"
ls -lh *_v3.sysmat 2>/dev/null || true

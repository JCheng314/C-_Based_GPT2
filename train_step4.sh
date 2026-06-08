#!/bin/bash
#SBATCH --job-name=gpt2_step4_mpi
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/step4_%j.out
#SBATCH --error=logs/step4_%j.err

set -e

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load NVIDIA HPC SDK / CUDA environment (provides OpenMPI with CUDA support)
ml course/cme213/nvhpc/24.1

# Make sure log directory exists
mkdir -p logs

# Compile Step 4
echo "Compiling train_step4.cu..."

nvcc -O3 -arch=sm_75 -ccbin=mpicxx \
    -Iinclude \
    src/train_step4.cu \
    src/linear_layers.cu \
    src/fused_layernorm.cu \
    src/backward_kernels.cu \
    src/flash_attention.cu \
    -o train_step4

echo "Compilation finished."

# Check dataset paths
if [ ! -f ./data/tinystories/train.bin ]; then
    echo "ERROR: ./data/tinystories/train.bin not found"
    exit 1
fi

if [ ! -f ./data/tinystories/val.bin ]; then
    echo "ERROR: ./data/tinystories/val.bin not found"
    exit 1
fi

# Run training
echo "Running Step 4 training with MPI..."

mpirun -np 2 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin "$@"

echo "Job finished."

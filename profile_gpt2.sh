#!/bin/bash
#SBATCH --job-name=gpt2_profile
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err

set -e

# Load modules
ml course/cme213/nvhpc/24.1

# Make sure logs dir exists
mkdir -p logs

echo "=== Compiling train_step4.cu ==="
nvcc -O3 -arch=sm_75 -ccbin=mpicxx \
    -Iinclude \
    src/train_step4.cu \
    src/linear_layers.cu \
    src/fused_layernorm.cu \
    src/backward_kernels.cu \
    src/flash_attention.cu \
    -o train_step4

echo "=== Running Profile: 1 GPU ==="
# Profile 1-GPU execution
# We trace cuda, mpi, and nvtx APIs
mpirun -np 1 nsys profile \
    --trace=cuda,mpi,osrt \
    -o logs/nsys_1gpu_rank%q{OMPI_COMM_WORLD_RANK} \
    -f true \
    ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin "$@"

echo "=== Running Profile: 2 GPUs ==="
# Profile 2-GPU execution (each rank gets its own profile)
mpirun -np 2 nsys profile \
    --trace=cuda,mpi,osrt \
    -o logs/nsys_2gpu_rank%q{OMPI_COMM_WORLD_RANK} \
    -f true \
    ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin "$@"

echo "=== Profiles generated successfully in logs/ ==="

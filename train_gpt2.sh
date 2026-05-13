#!/bin/bash
#SBATCH --job-name=train_step1
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_step1_%j.out
#SBATCH --error=logs/train_step1_%j.err

set -e

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run from the directory where you called sbatch
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

ml course/cme213/nvhpc/24.1

echo "Checking data files..."
ls -lh ./data/tinystories/train.bin
ls -lh ./data/tinystories/val.bin

echo "Compiling train_step1..."

nvcc -O3 -arch=sm_75 \
  -Iinclude \
  src/train_step1.cu \
  src/linear_layers.cu \
  src/backward_kernels.cu \
  -o train_step1

echo "Running train_step1..."

./train_step1 ./data/tinystories/train.bin ./data/tinystories/val.bin

echo "Job finished."
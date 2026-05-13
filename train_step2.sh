#!/bin/bash
#SBATCH --job-name=train_step2
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_step2_%j.out
#SBATCH --error=logs/train_step2_%j.err

set -e

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ml course/cme213/nvhpc/24.1

echo "Checking data files..."
ls -lh ./data/tinystories/train.bin
ls -lh ./data/tinystories/val.bin

echo "Compiling train_step2..."

nvcc -O3 -arch=sm_75 \
  -Iinclude \
  src/train_step2.cu \
  src/linear_layers.cu \
  src/fused_layernorm.cu \
  src/backward_kernels.cu \
  -o train_step2

echo "Running train_step2..."

./train_step2 ./data/tinystories/train.bin ./data/tinystories/val.bin

echo "Job finished."
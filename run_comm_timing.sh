#!/bin/bash
#SBATCH --job-name=gpt2_comm_timing
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/comm_timing_%j.out
#SBATCH --error=logs/comm_timing_%j.err

set -e

# Load modules
ml course/cme213/nvhpc/24.1

# Make sure dirs exist
mkdir -p logs benchmark_results

echo "=== Communication vs Compute Timing Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# ==================================================================
# Step 1: Recompile with the CUDA event instrumentation
# ==================================================================
echo "--- Compiling train_step4.cu (with timing instrumentation) ---"
nvcc -O3 -arch=sm_75 -ccbin=mpicxx \
    -Iinclude \
    src/train_step4.cu \
    src/linear_layers.cu \
    src/fused_layernorm.cu \
    src/backward_kernels.cu \
    src/flash_attention.cu \
    -o train_step4

echo "Compilation complete."
echo ""

# ==================================================================
# Step 2: Run 1-GPU with timing breakdown (200 steps for speed)
# ==================================================================
echo "--- Running 1-GPU timing (200 steps) ---"
mpirun -np 1 ./train_step4 \
    ./data/tinystories/train.bin ./data/tinystories/val.bin \
    --max-steps 200 --eval-every 200 \
    | tee benchmark_results/timing_1gpu.txt

echo ""

# ==================================================================
# Step 3: Run 2-GPU with timing breakdown (200 steps)
# ==================================================================
echo "--- Running 2-GPU timing (200 steps) ---"
mpirun -np 2 ./train_step4 \
    ./data/tinystories/train.bin ./data/tinystories/val.bin \
    --max-steps 200 --eval-every 200 \
    | tee benchmark_results/timing_2gpu.txt

echo ""
echo "=== Comm vs Compute timing complete ==="
echo "Results saved in benchmark_results/timing_{1,2}gpu.txt"

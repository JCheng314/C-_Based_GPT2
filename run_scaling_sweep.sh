#!/bin/bash
#SBATCH --job-name=gpt2_scaling
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err

set -e

# Load modules
ml course/cme213/nvhpc/24.1

# Make sure dirs exist
mkdir -p logs benchmark_results

echo "=== Batch Size Scaling Sweep (1 GPU) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# NOTE: Since B is a compile-time constant in train_step4.cu,
# we need to recompile for each batch size.
# We compile a special version that overrides B via a -D flag.

RESULTS_FILE="benchmark_results/scaling_sweep.csv"
echo "B,T,ms_per_step,total_time_ms,compute_ms,comm_ms,optim_ms" > "$RESULTS_FILE"

for BATCH_SIZE in 2 4 8 16; do
    echo ""
    echo "--- Compiling with B=${BATCH_SIZE} ---"

    # Create a temp source that overrides B
    # We use sed to replace the static const int B = 2; line
    TEMP_SRC="src/train_step4_B${BATCH_SIZE}.cu"
    sed "s/^static const int B = 2;$/static const int B = ${BATCH_SIZE};/" \
        src/train_step4.cu > "$TEMP_SRC"

    nvcc -O3 -arch=sm_75 -ccbin=mpicxx \
        -Iinclude \
        "$TEMP_SRC" \
        src/linear_layers.cu \
        src/fused_layernorm.cu \
        src/backward_kernels.cu \
        src/flash_attention.cu \
        -o "train_step4_B${BATCH_SIZE}" 2>&1

    echo "  Compiled train_step4_B${BATCH_SIZE}"

    echo "--- Running B=${BATCH_SIZE}, 200 steps ---"
    OUTPUT=$(mpirun -np 1 "./train_step4_B${BATCH_SIZE}" \
        ./data/tinystories/train.bin ./data/tinystories/val.bin \
        --max-steps 200 --eval-every 200 2>&1)

    echo "$OUTPUT"

    # Extract timing from the output
    AVG_STEP=$(echo "$OUTPUT" | grep "Average time per step" | awk '{print $NF}')
    TOTAL_TIME=$(echo "$OUTPUT" | grep "Total training loop time" | awk '{print $(NF-1)}')
    COMPUTE_MS=$(echo "$OUTPUT" | grep "Compute (fwd+bwd)" | awk '{print $3}')
    COMM_MS=$(echo "$OUTPUT" | grep "Communication:" | awk '{print $2}')
    OPTIM_MS=$(echo "$OUTPUT" | grep "Optimizer (AdamW)" | awk '{print $3}')

    echo "${BATCH_SIZE},64,${AVG_STEP},${TOTAL_TIME},${COMPUTE_MS},${COMM_MS},${OPTIM_MS}" >> "$RESULTS_FILE"

    # Cleanup temp files
    rm -f "$TEMP_SRC" "train_step4_B${BATCH_SIZE}"
done

echo ""
echo "=== Scaling sweep complete ==="
echo "Results:"
cat "$RESULTS_FILE"

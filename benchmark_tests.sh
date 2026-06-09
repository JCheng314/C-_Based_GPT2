#!/bin/bash
#SBATCH --job-name=gpt2_bench
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -e

# Load modules
ml course/cme213/nvhpc/24.1

# Make sure logs/benchmark dirs exist
mkdir -p logs benchmark_results tmp
export TMPDIR="$PWD/tmp"

echo "=== Running Benchmark Tests ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# ==================================================================
# 1. Per-kernel timing table from existing nsys profiles
# ==================================================================
echo ""
echo "--- [1/7] Extracting per-kernel summaries from nsys profiles ---"

nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
     nsys_1gpu_rank0.nsys-rep > benchmark_results/kern_summary_1gpu.csv 2>&1 || true

nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
     nsys_2gpu_rank0.nsys-rep > benchmark_results/kern_summary_2gpu.csv 2>&1 || true

echo "  -> kern_summary_1gpu.csv, kern_summary_2gpu.csv"

# ==================================================================
# 2. MPI call breakdown (try multiple report names for compatibility)
# ==================================================================
echo ""
echo "--- [2/7] Extracting MPI summary ---"

# Try different report names — nsys versions vary
if nsys stats --report mpi_gpu_sum --format csv --force-export=true \
     nsys_2gpu_rank0.nsys-rep > benchmark_results/mpi_summary.csv 2>&1; then
    echo "  -> mpi_summary.csv (via mpi_gpu_sum)"
elif nsys stats --report mpi_sum --format csv --force-export=true \
     nsys_2gpu_rank0.nsys-rep > benchmark_results/mpi_summary.csv 2>&1; then
    echo "  -> mpi_summary.csv (via mpi_sum)"
else
    echo "  WARNING: No MPI report available. Listing available reports:"
    nsys stats --help nsys_2gpu_rank0.nsys-rep 2>&1 | head -40 || true
fi

# ==================================================================
# 3. CUDA API call overhead (launch + sync timing)
# ==================================================================
echo ""
echo "--- [3/7] Extracting CUDA API call summary ---"

nsys stats --report cuda_api_sum --format csv --force-export=true \
     nsys_1gpu_rank0.nsys-rep > benchmark_results/cuda_api_sum_1gpu.csv 2>&1 || true

nsys stats --report cuda_api_sum --format csv --force-export=true \
     nsys_2gpu_rank0.nsys-rep > benchmark_results/cuda_api_sum_2gpu.csv 2>&1 || true

echo "  -> cuda_api_sum_1gpu.csv, cuda_api_sum_2gpu.csv"

# ==================================================================
# 4. GPU hardware specs (for roofline calculations)
# ==================================================================
echo ""
echo "--- [4/7] Collecting GPU hardware specs ---"

nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.current,pcie.link.width.current \
           --format=csv | tee benchmark_results/gpu_specs.csv

echo ""
echo "  GPU specs saved to benchmark_results/gpu_specs.csv"

# ==================================================================
# 5. Nsight Compute: cross_entropy_forward_kernel (60.7% of GPU time)
# ==================================================================
echo ""
echo "--- [5/7] NCU profiling: cross_entropy_forward_kernel ---"

# Run with mpirun -np 1 to profile single-GPU execution for cleaner metrics
# Limit to first invocation (--launch-count 1) to keep runtime reasonable
mpirun -np 1 ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  --kernel-name cross_entropy_forward_kernel \
  --launch-count 3 \
  ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin \
  --max-steps 5 --eval-every 100 \
  > benchmark_results/ncu_cross_entropy.txt 2>&1

echo "  -> ncu_cross_entropy.txt"

# ==================================================================
# 6. Nsight Compute: linear_backward_dA_kernel (24.6% of GPU time)
# ==================================================================
echo ""
echo "--- [6/7] NCU profiling: linear_backward_dA_kernel ---"

mpirun -np 1 ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  --kernel-name linear_backward_dA_kernel \
  --launch-count 3 \
  ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin \
  --max-steps 5 --eval-every 100 \
  > benchmark_results/ncu_linear_backward_dA.txt 2>&1

echo "  -> ncu_linear_backward_dA.txt"

# ==================================================================
# 7. Nsight Compute: hierarchical_gemm_2x4_kernel (forward GEMM)
# ==================================================================
echo ""
echo "--- [7/7] NCU profiling: hierarchical_gemm_2x4_kernel ---"

mpirun -np 1 ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  --kernel-name hierarchical_gemm_2x4_kernel \
  --launch-count 3 \
  ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin \
  --max-steps 5 --eval-every 100 \
  > benchmark_results/ncu_gemm.txt 2>&1

echo "  -> ncu_gemm.txt"

echo ""
echo "=== All benchmark tests complete ==="
echo "Results in benchmark_results/"
ls -la benchmark_results/
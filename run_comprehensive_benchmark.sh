#!/bin/bash
#SBATCH --job-name=comprehensive_benchmark
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -e

echo "Starting comprehensive benchmarking job..."
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load modules
ml course/cme213/nvhpc/24.1

# Clean and compile
make clean
make comprehensive_benchmark
make train_step4

mkdir -p results
mkdir -p figures

# Clear old results to prevent mixing runs
rm -f results/req8_allreduce.csv
touch results/req8_allreduce.csv

# ==========================================================
# 1. Single-GPU & Multi-GPU AllReduce Sweeps
# ==========================================================
echo "Running single-GPU sweeps & 1-rank AllReduce..."
mpirun -np 1 ./comprehensive_benchmark

echo "Running 2-rank AllReduce..."
mpirun -np 2 ./comprehensive_benchmark

echo "Running 4-rank AllReduce..."
mpirun -np 4 ./comprehensive_benchmark

# ==========================================================
# 2. MPI Strong Scaling Sweeps (Req 8)
# ==========================================================
echo "Running MPI Strong Scaling Sweeps..."
# Keep a backup of train_step4.cu
cp src/train_step4.cu src/train_step4.cu.bak

# Total batch size B_total = 4
# Rank 1: B = 4
echo "Running strong scaling: 1 Rank (B=4)..."
sed -i 's/static const int B = .*/static const int B = 4;/g' src/train_step4.cu
rm -f train_step4
make train_step4
mpirun -np 1 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/strong_1.log

# Rank 2: B = 2
echo "Running strong scaling: 2 Ranks (B=2)..."
sed -i 's/static const int B = .*/static const int B = 2;/g' src/train_step4.cu
rm -f train_step4
make train_step4
mpirun -np 2 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/strong_2.log

# Rank 4: B = 1
echo "Running strong scaling: 4 Ranks (B=1)..."
sed -i 's/static const int B = .*/static const int B = 1;/g' src/train_step4.cu
rm -f train_step4
make train_step4
mpirun -np 4 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/strong_4.log

# Restore from backup for weak scaling section (don't delete .bak yet)
cp src/train_step4.cu.bak src/train_step4.cu

# Parse strong scaling times
t_s1=$(grep "Average time per step:" results/strong_1.log | awk '{print $5}')
t_s2=$(grep "Average time per step:" results/strong_2.log | awk '{print $5}')
t_s4=$(grep "Average time per step:" results/strong_4.log | awk '{print $5}')

# Compute speedup and efficiency
s_1=1.0
e_1=1.0

s_2=$(awk "BEGIN {print $t_s1 / $t_s2}")
e_2=$(awk "BEGIN {print $s_2 / 2.0}")

s_4=$(awk "BEGIN {print $t_s1 / $t_s4}")
e_4=$(awk "BEGIN {print $s_4 / 4.0}")

echo "ranks,step_time_ms,speedup,efficiency" > results/req8_mpi_strong.csv
echo "1,$t_s1,$s_1,$e_1" >> results/req8_mpi_strong.csv
echo "2,$t_s2,$s_2,$e_2" >> results/req8_mpi_strong.csv
echo "4,$t_s4,$s_4,$e_4" >> results/req8_mpi_strong.csv

# ==========================================================
# 3. MPI Weak Scaling Sweeps (Req 8)
# ==========================================================
echo "Running MPI Weak Scaling Sweeps (Per-rank B=2)..."

# Set per-rank B=2 for all weak scaling runs (workload scales WITH rank count)
sed -i 's/static const int B = .*/static const int B = 2;/g' src/train_step4.cu
rm -f train_step4
make train_step4

echo "Running weak scaling: 1 Rank..."
mpirun -np 1 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/weak_1.log

echo "Running weak scaling: 2 Ranks..."
mpirun -np 2 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/weak_2.log

echo "Running weak scaling: 4 Ranks..."
mpirun -np 4 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 50 > results/weak_4.log

# Final restore: delete backup, rebuild with original source
cp src/train_step4.cu.bak src/train_step4.cu
rm -f src/train_step4.cu.bak
rm -f train_step4
make train_step4

# Parse weak scaling times
t_w1=$(grep "Average time per step:" results/weak_1.log | awk '{print $5}')
t_w2=$(grep "Average time per step:" results/weak_2.log | awk '{print $5}')
t_w4=$(grep "Average time per step:" results/weak_4.log | awk '{print $5}')

# Compute weak scaling efficiency (T1 / Tn) and speedup (n * T1 / Tn)
eff_w1=1.0
spd_w1=1.0

eff_w2=$(awk "BEGIN {print $t_w1 / $t_w2}")
spd_w2=$(awk "BEGIN {print 2.0 * $eff_w2}")

eff_w4=$(awk "BEGIN {print $t_w1 / $t_w4}")
spd_w4=$(awk "BEGIN {print 4.0 * $eff_w4}")

echo "ranks,step_time_ms,speedup,efficiency" > results/req8_mpi_weak.csv
echo "1,$t_w1,$spd_w1,$eff_w1" >> results/req8_mpi_weak.csv
echo "2,$t_w2,$spd_w2,$eff_w2" >> results/req8_mpi_weak.csv
echo "4,$t_w4,$spd_w4,$eff_w4" >> results/req8_mpi_weak.csv

# ==========================================================
# 4. Multi-Rank Loss Curve Consistency Check (Req 9)
# ==========================================================
echo "Running Loss Curve Consistency Check..."
# Use --eval-every 10 so we get ~20 data points for the plot
echo "Running 1 rank training (200 steps, eval every 10)..."
mpirun -np 1 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 200 --eval-every 10 > results/loss_1rank.log

echo "Running 2 rank training (200 steps, eval every 10)..."
mpirun -np 2 ./train_step4 ./data/tinystories/train.bin ./data/tinystories/val.bin --max-steps 200 --eval-every 10 > results/loss_2rank.log

# ==========================================================
# 5. Plotting (Req 7, 8, 9)
# ==========================================================
echo "Running Python plotting script..."
/home/cme213/stephone/miniconda3/bin/python scripts/plot_results.py

echo "Job finished successfully. All plots and results are in results/ and figures/ directories."

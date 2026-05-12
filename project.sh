#!/bin/bash
#SBATCH --job-name=gpt2_benchmark
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-pascal

./gpt2_benchmark
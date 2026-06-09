#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

set -e

echo "Starting data preparation..."
/home/cme213/stephone/miniconda3/bin/python prepare_tinystories.py
echo "Data preparation complete."

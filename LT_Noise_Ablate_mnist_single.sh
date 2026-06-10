#!/bin/bash
#SBATCH --job-name=LT_Noise_MNIST_single-class
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/LT_Noise_MNIST_single-class_%j.out
#SBATCH --error=slurm_output/LT_Noise_MNIST_single-class_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=END

set -euo pipefail

mkdir -p slurm_output
mkdir -p results

# --- Env ---
module load python
source ./envs/ssl_env/bin/activate




echo ">>> Running experiments..."

python LT_Noise.py --dataset mnist --epochs 10 --model smallcnn --imbalance_mode single_class --output_csv mnist_single-class_ablate --imbalance_factors 1.0 0.75 0.5 0.25 0.1 0.05 0.01 0.005 --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20


echo "[INFO] Job complete."
#!/bin/bash
#SBATCH --job-name=LT_Noise_MNIST
#SBATCH --account=def-pfieguth
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/LT_Noise_MNIST_%j.out
#SBATCH --error=slurm_output/LT_Noise_MNIST_%j.err


set -euo pipefail

mkdir -p slurm_output
mkdir -p results

# --- Env ---
module load python
source ./envs/ssl_env/bin/activate




echo ">>> Running experiments..."

python LT_Noise.py --dataset mnist --epochs 10 --model smallcnn --output_csv mnist_smallcnn_ablate --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20


echo "[INFO] Job complete."
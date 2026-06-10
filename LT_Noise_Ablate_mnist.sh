#!/bin/bash
#SBATCH --job-name=LT_Noise_MNIST
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/LT_Noise_MNIST_%j.out
#SBATCH --error=slurm_output/LT_Noise_MNIST_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=END

set -euo pipefail

mkdir -p slurm_output
mkdir -p results

# --- Env ---
module load python
source ./envs/ssl_env/bin/activate




echo ">>> Running experiments..."

python LT_Noise.py --dataset mnist --epochs 10 --model smallcnn --output_csv mnist_smallcnn_ablate_s50 --seeds 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50


echo "[INFO] Job complete."
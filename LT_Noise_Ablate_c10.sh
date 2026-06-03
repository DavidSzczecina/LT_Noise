#!/bin/bash
#SBATCH --job-name=LT_Noise_Ablate_c10
#SBATCH --account=def-pfieguth
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/LT_Noise_Ablate_c10_%j.out
#SBATCH --error=slurm_output/LT_Noise_Ablate_c10_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL

set -euo pipefail

mkdir -p slurm_output
mkdir -p results

# --- Env ---
module load python
source ./envs/ssl_env/bin/activate




echo ">>> Running experiments..."

python LT_noise.py --dataset cifar10 --epochs 15 --seeds 6 7 8 9 10


echo "[INFO] Job complete."
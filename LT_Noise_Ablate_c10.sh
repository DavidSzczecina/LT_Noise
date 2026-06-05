#!/bin/bash
#SBATCH --job-name=LT_Noise_Ablate_c10
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
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

python LT_Noise.py --dataset cifar10 --epochs 15 --model resnet18 --exp_name cifar10 --output_csv cifar10_resnet18_ablate --seeds 1 2 3 4 5 6 7 8 9 10


echo "[INFO] Job complete."
#!/bin/bash
#SBATCH --job-name=LT_Noise_Ablate_c100
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/LT_Noise_Ablate_c100_%j.out
#SBATCH --error=slurm_output/LT_Noise_Ablate_c100_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=END

set -euo pipefail

mkdir -p slurm_output
mkdir -p results

# --- Env ---
module load python
source ./envs/ssl_env/bin/activate




echo ">>> Running experiments..."

python LT_Noise.py --dataset cifar100 --epochs 15 --model resnet18 --output_csv cifar100_resnet18 --seeds 1 2 3 4 5 6 7 8 9 10


echo "[INFO] Job complete."
#!/bin/bash
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 8GB
#SBATCH -p short
#SBATCH -t 4:00:00

module load rocky8 micromamba
micromamba activate openamp

python run_uni_amp_balanced.py
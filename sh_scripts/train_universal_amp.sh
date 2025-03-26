#!/bin/bash
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 16
#SBATCH --mem-per-gpu 32GB
#SBATCH -p gpu
#SBATCH -t 96:00:00

module load rocky8 micromamba
micromamba activate openamp

python train_universal_amp.py --num_workers 16 --train_batch_size 32 --val_batch_size 24
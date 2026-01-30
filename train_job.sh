#!/bin/bash
#SBATCH --job-name=fer_training
#SBATCH --partition=NvidiaAll
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


set -euo pipefail


source ~/miniconda3/etc/profile.d/conda.sh
conda activate FER-env

cd /home/v/vahutinskij/work/fer_project


which python
python --version
nvidia-smi


python run_training.py


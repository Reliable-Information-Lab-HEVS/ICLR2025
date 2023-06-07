#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=10000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Verify working directory
echo $(pwd)

#nvidia-smi

nvcc --version

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

cat /usr/local/cuda/version.txt

# python3 hierarchical.py "$@"

conda deactivate
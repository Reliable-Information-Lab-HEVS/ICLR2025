#!/bin/bash

#SBATCH --job-name=memory_estimator
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:6
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

python3 memory_estimator.py "$@"

conda deactivate
#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
# conda activate llm-playground
conda activate test

python3 test2.py "$@"

conda deactivate
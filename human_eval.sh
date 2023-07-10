#!/bin/bash

#SBATCH --job-name=humaneval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:8
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

python3 human_eval.py

conda deactivate

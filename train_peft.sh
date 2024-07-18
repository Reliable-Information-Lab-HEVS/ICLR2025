#!/bin/bash

#SBATCH --job-name=train_peft
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:v100:2
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

python3 train_peft.py

conda deactivate

#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=21G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:v100:3
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

python3 -u test_allocator.py
# srun --exclusive --exact -v --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --mem=5G python3 -u test_allocator.py

conda deactivate

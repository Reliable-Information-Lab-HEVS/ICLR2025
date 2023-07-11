#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:v100:3
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Verify working directory
echo $(pwd)

nvidia-smi

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

python3 test.py "$@"

conda deactivate
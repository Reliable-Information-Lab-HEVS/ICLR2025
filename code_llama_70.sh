#!/bin/bash

#SBATCH --job-name=aatk_english_single
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=100G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:5
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm-playground


python3 -u AATK_english.py code-llama-70B-instruct "$@"

conda deactivate

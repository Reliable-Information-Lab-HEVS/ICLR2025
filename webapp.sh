#!/bin/bash

#SBATCH --job-name=webapp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:3
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# Note: the -u option is absolutely necesary here to force the flush of the link 
# to connect to the app!
python3 -u webapp.py

conda deactivate

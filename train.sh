#!/bin/bash

#SBATCH --job-name=train_walliser
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# Needs to be launched in this way to correctly use Torch DDP
python -m torch.distributed.launch --nproc_per_node 4 train.py

conda deactivate

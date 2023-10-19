#!/bin/bash

#SBATCH --job-name=aatk
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=200G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:5
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# make sure the job scheduler is using very few resources that are not the same as the subprocesses
# that will be launched
# srun --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --mem=5G python3 -u AATK_wrapper.py "$@"

# How are the resources of this python script managed? Because we can still spawn child processes correctly
# even with exclusive option
python3 -u AATK_wrapper.py "$@"

conda deactivate

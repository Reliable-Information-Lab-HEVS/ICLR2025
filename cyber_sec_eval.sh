#!/bin/bash

#SBATCH --job-name=cyber_sec_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=230G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:6
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# make sure the job scheduler is using very few resources that are not the same as the subprocesses
# that will be launched
# srun --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --mem=5G python3 -u AATK_instruct_wrapper.py "$@"

python3 -u cyber_sec_eval_wrapper.py "$@"

conda deactivate

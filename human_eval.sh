#!/bin/bash

#SBATCH --job-name=humaneval
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=200G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:h200:2
#SBATCH --chdir=/cluster/raid/home/stea/ICLR2025
#SBATCH --qos=normal
echo "Running human eval benchmark"

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# show what conda env is active
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# make sure the job scheduler is using very few resources that are not the same as the subprocesses
# that will be launched
# srun --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --mem=5G python3 -u human_eval_wrapper.py "$@"

# How are the resources of this python script managed? Because we can still spawn child processes correctly
# even with exclusive option
echo "Executing the command: python3 -u human_eval_wrapper.py $@"
python3 -u human_eval_wrapper.py "$@"

conda deactivate

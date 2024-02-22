#!/bin/bash

#SBATCH --job-name=experiment_app
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:5
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# Launch frpc server - note that the '&' is essential to run the command in a non-blocking way
# srun --exclusive --exact --ntasks=1 --cpus-per-task=1 --mem=1G ../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc.toml &
../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc.toml &
# Launch app
python3 experiment_webapp.py

conda deactivate

#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:0
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate test

# srun --exclusive --exact --ntasks=1 --cpus-per-task=1 --mem=1G ../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc.toml &
../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc.toml &

# ss -tulpn
python3 foo.py 
# ss -tulpn





conda deactivate

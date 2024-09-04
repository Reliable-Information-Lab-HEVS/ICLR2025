#!/bin/bash

#SBATCH --job-name=cybersec-evaluate
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=60
#SBATCH --mem=80G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# python3 -u cyber_sec_evaluate_wrapper.py "$@"
# python3 -u cyber_sec_evaluate.py "$@"
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/star-chat-alpha/float16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/code-llama-34B-instruct/bfloat16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/llama3-8B-instruct/bfloat16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/command-r/bfloat16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/zephyr-7B-beta/bfloat16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/mistral-7B-instruct-v2/bfloat16/temperature_0.2.jsonl --workers 16 &
srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task=8 --mem=10G python3 -u cyber_sec_evaluate.py /cluster/raid/home/vacy/LLMs/results/CyberSecEval_instruct_llama3_python/completions/starling-7B-beta/bfloat16/temperature_0.2.jsonl --workers 16 &

conda deactivate

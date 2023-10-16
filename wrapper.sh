#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
srun --ntasks 1 --gpus 1 --cpus-per-task 2 --mem 20G python3 test_srun.py
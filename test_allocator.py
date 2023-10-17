import subprocess
import time
import os
import sys
import torch
import engine
from helpers import utils

command = 'srun --ntasks=1 --gpus-per-task={gpus} --cpus-per-task=2 --mem=60G python3 test_srun.py'

# print(os.environ['CUDA_VISIBLE_DEVICES'])


t0 = time.time()
# num_gpus = torch.cuda.device_count()
num_gpus = 5

models = ['bloom-560M']*5

print(f'Launching computations with {num_gpus} gpus available.')

# gpu_footprints = engine.estimate_number_of_gpus(models, False, False)
gpu_footprints = [1]*5
commands = [f'--exact -v python3 -u test_srun.py' for _ in models]

utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands, cpus_per_task=1, memory=3)

dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')
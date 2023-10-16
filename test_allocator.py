import subprocess
import time
import os
import sys
import torch
import engine
from helpers import utils

print(os.environ['CUDA_VISIBLE_DEVICES'])

command = 'srun --ntasks=1 --gpus-per-task={gpus} --cpus-per-task=2 --mem=60G python3 test_srun.py'

# t0 = time.time()
# processes = []
# for i in range(5):
#     # subprocess.run(command.format(gpus=1).split(' '))
#     # p = subprocess.Popen([os.path.join(utils.ROOT_FOLDER, 'wrapper.sh'), f'{i}'])
#     p = subprocess.Popen(command.format(gpus=1).split(' '), stdout=sys.stdout, stderr=sys.stderr)
#     # p = subprocess.Popen(['ls'], stdout=sys.stdout, stderr=sys.stderr)
#     print(f'Is running') if p.poll() is None else print('Is not running')
#     processes.append(p)


# for p in processes:
#     while p.poll() is None:
#         time.sleep(1)

# dt = time.time() - t0
# print(f'Everything done in {dt:.2f} s')


t0 = time.time()
num_gpus = torch.cuda.device_count()

# Select models (only keep the good coders)
small_models = engine.SMALL_GOOD_CODERS
large_models = engine.LARGE_GOOD_CODERS
models = small_models + large_models

print(f'Launching computations with {num_gpus} gpus available.')

gpu_footprints = engine.estimate_number_of_gpus(models, False, False)
commands = [f'echo "starting" & sleep 30' for _ in models]
utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)
dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')
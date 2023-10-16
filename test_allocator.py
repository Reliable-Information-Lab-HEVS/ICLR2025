import subprocess
import time
import os

from helpers import utils

print(os.environ['CUDA_VISIBLE_DEVICES'])

command = 'srun --ntasks 1 --gpus 1 --cpus-per-task 2 --mem 20G python3 test_srun.py'

t0 = time.time()
for i in range(3):
    # subprocess.run(command.split(' '))
    p = subprocess.Popen([os.path.join(utils.ROOT_FOLDER, 'wrapper.sh'), f'{i}'])
dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')
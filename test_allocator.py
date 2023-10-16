import subprocess
import time
import os
import sys

from helpers import utils

print(os.environ['CUDA_VISIBLE_DEVICES'])

command = 'srun --ntasks=1 --gpus-per-task={gpus} --cpus-per-task=2 --mem=20G python3 test_srun.py'

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
utils.dispatch_jobs_srun([1]*5, 3, ['python3 test_srun.py']*5)
dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')
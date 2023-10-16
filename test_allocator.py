import subprocess
import time

command = 'srun --gpus 1 --cpus-per-task 2 --mem 20G python3 test_srun.py'

t0 = time.time()
for i in range(3):
    subprocess.run(command.split(' '))
    # p = subprocess.run(['wrapper.sh', f'{i}'])
dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')
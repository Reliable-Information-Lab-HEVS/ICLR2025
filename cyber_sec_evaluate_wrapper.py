import time
import argparse
import subprocess
import shlex
import sys

from helpers import cybersec


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CyberSec Eval')
    parser.add_argument('--workers', type=int, default=16, help='Number of workers to use.')

    args = parser.parse_args()
    workers = args.workers

    for dataset in cybersec.DATASETS:
        t0 = time.time()
        try:
            files = cybersec.extract_filenames(dataset=dataset, category='completions', only_unprocessed=True)
        except RuntimeError:
            continue

        # Create the commands to run
        commands = [f'python3 -u cyber_sec_evaluate.py {file} --workers {workers}' for file in files]

        # Commands to run on slrum
        full_commands = [
            (f'srun --exclusive --exact --ntasks=1 --gpus-per-task=0 --cpus-per-task={workers // 2} '
                            f'--mem={5}G {executable}') for executable in commands
        ]

        processes = []
        for full_command in full_commands:
            p = subprocess.Popen(shlex.split(full_command), stdout=sys.stdout, stderr=sys.stderr, bufsize=0)
            processes.append(p)

        # Start infinite loop until all processes are finished
        while True:
            # Find the indices of the processes that are finished if any
            indices_to_remove = []
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    # the wait() is only used to clean the subprocess (avoid zombies), as it is already done at this point
                    process.wait()
                    indices_to_remove.append(i)

            if not len(indices_to_remove) == 0:
                # Remove processes which are done
                processes = [process for i, process in enumerate(processes) if i not in indices_to_remove]

            #  If all jobs are finished, break from the infinite loop
            if len(processes) == 0:
                break

            # Sleep before restarting the loop
            time.sleep(30)
                

        dt = time.time() - t0
        print(f'Overall it took {dt/3600:.2f}h !')


import multiprocessing as mp
import queue as q
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os

from helpers import utils

# queue = mp.Queue()

# for i in range(40):
#     queue.put_nowait(i)


# def main(rank, queue):

#     while True:

#         try:
#             model = queue.get_nowait()
#         except q.Empty:
#             return
        
#         sleep_time = 10 if rank == 0 else 1
#         print(f'Rank {rank}: time is {datetime.now().isoformat()}. Sleeping for {sleep_time} s')
#         time.sleep(sleep_time)
#         print(f'Rank {rank}: time is {datetime.now().isoformat()}. Done sleeping!')


# if __name__ == '__main__':

#     num_workers = 4

#     mp.spawn(main, args=(queue,), nprocs=num_workers, join=True)

#     print('Main process done!')


foo = [i for i in range(40)]


def main(useless, sleep_time = 2):

    # print(type(mp.current_process()))
    # print(isinstance(mp.current_process(), mp.context.SpawnProcess))
    name = mp.current_process().name
    rank = int(name[-1]) - 1
        
    sleep_time = 10 if rank == 0 else sleep_time
    print(f'Rank {rank}: time is {datetime.now().isoformat()}. Sleeping for {sleep_time} s')
    time.sleep(sleep_time)
    print(f'Rank {rank}: time is {datetime.now().isoformat()}. Done sleeping!')


def test(useless):

    time.sleep(5)
    rank = utils.find_rank_of_subprocess_inside_the_pool()
    print(f"Process {rank}: visible device: {os.environ['CUDA_VISIBLE_DEVICES']}")


if __name__ == '__main__':

    with ProcessPoolExecutor(max_workers=3, initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        pool.map(test, [1,1,1], chunksize=1)

    print(f"Main process: {os.environ['CUDA_VISIBLE_DEVICES']}")
import multiprocessing as mp
import time
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import torch

import engine
from engine import stopping
from engine import loader
from engine.prompt_template import PROMPT_MODES
from engine.code_parser import CodeParser, PythonParser
from helpers import datasets
from helpers import utils

from transformers import PreTrainedModel

@utils.duplicate_function_for_gpu_dispatch
def target(name: str, foo, bar = 3):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    # model = engine.HFModel(name)
    # print(f'Gpus as seen by torch: {model.get_gpu_devices()}')


@utils.duplicate_function_for_gpu_dispatch
def sleep(dt: float = 4):
    time.sleep(dt)
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')


if __name__ == '__main__':

    # num_gpus = torch.cuda.device_count()
    num_gpus = 3
    gpu_footprints = [1,1,1,2,2,1,3]

    t1 = time.time()
    utils.dispatch_jobs(gpu_footprints, num_gpus, sleep)
    dt1 = time.time() - t1

    
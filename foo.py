import multiprocessing as mp
import time
import os
from concurrent.futures import ProcessPoolExecutor

import torch

# from engine import loader
from helpers import utils


def test():
     print('This is a test')

@utils.duplicate_function_for_gpu_dispatch
def target(name: str, foo, bar = 3):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    time.sleep(5)
    print('Done!')
    test()


@utils.duplicate_function_for_gpu_dispatch
def target2(name: str, foo, bar = 3):
    time.sleep(5)
    print('target2')
    test()



LARGE_MODELS = (
    'gpt-neoX-20B',
    # 'opt-30B',
    'llama2-70B',
)


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()

    # model_footprints = []
    # # Estimate number of gpus needed for each model
    # for model in LARGE_MODELS:
    #     quantization = model == 'bloom-176B'
    #     gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
    #     model_footprints.append(gpu_needed)

    model_footprints = [2,5]


    args = ([1,2],)
    utils.dispatch_jobs(LARGE_MODELS, model_footprints, num_gpus, target, args)
    
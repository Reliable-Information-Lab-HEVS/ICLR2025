import multiprocessing as mp
import time
import os
import torch
from concurrent.futures import ProcessPoolExecutor

from engine import loader
from helpers import utils

def test():
     print('This is a test')

@utils.duplicate_function_for_gpu_dispatch
def target(foo, bar):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    time.sleep(5)
    print('Done!')
    test()



LARGE_MODELS = (
    'gpt-neoX-20B',
    'llama2-70B',
)


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()

    model_footprints = []
    # Estimate number of gpus needed for each model
    for model in LARGE_MODELS:
        quantization = model == 'bloom-176B'
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
        model_footprints.append(gpu_needed)

    utils.dispatch_jobs(LARGE_MODELS, model_footprints, num_gpus, target, [1,2], [3,4])
    # dispatch_jobs(LARGE_MODELS, num_gpus, utils.target_gpu_dispatch, [1,2], [3,4])
    # dispatch_jobs(LARGE_MODELS, num_gpus, target_func_on_gpu, [1,2], [3,4])

    # with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn'),
    #                          initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        
    #     _ = list(pool.map(target, LARGE_MODELS, LARGE_MODELS, chunksize=1))
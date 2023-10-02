import torch
import time

import engine
from helpers import utils


@utils.duplicate_function_for_gpu_dispatch
def test(model):
    try:
        foo = engine.HFModel(model)
    except Exception as e:
        print(f'Issue with {model}: {type(e).__name__}: {str(e)}')
        pass

if __name__ == '__main__':
    models = engine.SMALL_MODELS
    gpu_footprints = engine.estimate_number_of_gpus(models, False, False)
    utils.dispatch_jobs(gpu_footprints, torch.cuda.device_count(), test, models)

# model = engine.HFModel('star-coder-plus')


import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math
import warnings
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

import engine
from engine import loader
from helpers import utils, datasets


@utils.duplicate_function_for_gpu_dispatch
def test():
    model_name = 'bloom-560M'
    model = engine.HFModel(model_name)

    dataset = datasets.HumanEval()

    t0 = time.time()
    for sample in dataset[0:10]:
        out = model(sample['prompt'], prompt_template_mode='generation')
    dt = time.time() - t0
    print(f'Done in {dt:.2f} s, with visible devices : {os.environ["CUDA_VISIBLE_DEVICES"]}')

# if __name__ == '__main__':

    # utils.dispatch_jobs([1]*5, 3, test)

test()

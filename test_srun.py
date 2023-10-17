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


def test():
    t0 = time.time()
    model_name = 'bloom-560M'
    print(f'Starting with {model_name}')
    model = engine.HFModel(model_name)

    dataset = datasets.HumanEval()
    for sample in dataset[0:30]:
        out = model(sample['prompt'], prompt_template_mode='generation')
    dt = time.time() - t0
    print(f'Done in {dt:.2f} s, with visible devices : {os.environ["CUDA_VISIBLE_DEVICES"]}')

if __name__ == '__main__': 
    test()

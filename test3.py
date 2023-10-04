import torch
import time

import engine
from engine import loader
from helpers import utils

model_name = 'bloom-176B'

model = engine.HFModel(model_name, quantization_8bits=True, max_fraction_gpu_0=0.6,
                       max_fraction_gpus=0.6)
print(model.get_gpu_memory_footprint())


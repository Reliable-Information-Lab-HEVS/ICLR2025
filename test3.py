import torch
import time

import engine
from engine import loader
from helpers import utils

model_name = 'bloom-176B'

model = engine.HFModel(model_name, quantization_8bits=True)
print(model.get_gpu_memory_footprint())


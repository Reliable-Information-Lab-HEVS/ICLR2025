import torch
import time

import engine
from engine import loader
from helpers import utils

model_name = 'bloom-176B'

min_gpu_needed, max_memory_map = loader.estimate_model_gpu_footprint(model_name)
print(f'Min gpu needed: {min_gpu_needed}')
model = engine.HFModel(model_name)
print(model.get_gpu_memory_footprint())


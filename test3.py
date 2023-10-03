import torch
import time

import engine
from helpers import utils


model = engine.HFModel('bloom-176B')
print(model.get_gpu_memory_footprint())


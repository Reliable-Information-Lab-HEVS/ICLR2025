import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math

import engine
from helpers import utils, datasets

model_name = 'llama2-7B'
model = engine.HFModel(model_name)

dataset = datasets.HumanEval()
prompt = dataset[0]['prompt']

out1 = model(prompt, max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
             stopping_patterns=True)

print(out1)
print('\n\n')

out2 = model(prompt.strip(), max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
             stopping_patterns=True)

print(out2)

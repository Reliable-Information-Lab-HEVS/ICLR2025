import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math
import warnings

from transformers import AutoModelForCausalLM

import engine
from engine import loader
from helpers import utils, datasets

# model_name = 'llama2-7B'

# t0 = time.time()
# model = engine.HFModel(model_name)
# dt = time.time() - t0
# print(f'Time for loading: {dt:.2f} s')

# dataset = datasets.HumanEval()
# prompt = dataset[0]['prompt']

# t0 = time.time()
# out1 = model(prompt, max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
#              stopping_patterns=True)

# print(out1)
# print('\n\n')

# out2 = model(prompt.strip(), max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
#              stopping_patterns=True)
# dt = time.time() - t0

# print(out2)

# print('\n\n')

# print(f'Time for 2 inferences: {dt:.2f} s')





# model_name = 'bloom-560M'

# model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m', low_cpu_mem_usage=True)
# model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m')

# model = load_model(model_name)

# model = engine.HFModel(model_name)
# print(model.dtype_category())
# print(model.get_gpu_memory_footprint())

# from transformers import AutoModelForCausalLM


# foo = loader.estimate_model_gpu_footprint('bloom-560M')


LARGE_MODELS = (
    'gpt-neoX-20B',
    'opt-30B',
    'opt-66B',
    'llama2-70B',
    'llama2-70B-chat',
    'bloom-176B',
)

model_footprints = []
for model in LARGE_MODELS:
    # Override quantization for bloom because it's too big
    if model == 'bloom-176B':
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=True,
                                                            quantization_4bits=False)
    else:
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model)
    model_footprints.append(gpu_needed)

print(model_footprints)
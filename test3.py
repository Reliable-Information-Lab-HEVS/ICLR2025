from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import time

import engine

# model_name = 'facebook/opt-13b'

# max_memory = {0: '3GiB', 1: '25GiB', 2:'25GiB'}
# device_map = 'balanced'
# t0 = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map,
#                                              max_memory=max_memory)
# dt0 = time.time() - t0
# print(f'Total time to load: {dt0:.2f} s')

# for i in range(torch.cuda.device_count()):
#     print(f'GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GiB')

# print(model.hf_device_map)





# t0 = time.time()
# config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)
#     model.tie_weights()
#     device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=torch.float16)
# dt0 = time.time() - t0
# print(f'Time to compute device_map: {dt0:.2f} s')

# t1 = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
# dt1 = time.time() - t1
# print(f'Time to load model: {dt1:.2f} s')

# print(f'Total time: {dt0 + dt1:.2f} s')

# for i in range(torch.cuda.device_count()):
#     print(f'GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GiB')

# print(model.hf_device_map)

# from engine import loader
# model_name = 'bloom-176B'
# # print(loader.estimate_model_gpu_footprint(model_name, quantization_8bits=True, max_fraction_gpu_0=0.9,
#                                         #   max_fraction_gpus=0.9))
# model = engine.HFModel(model_name, quantization_8bits=True, max_fraction_gpu_0=0.9, max_fraction_gpus=0.9)
# print(model.get_gpu_memory_footprint())

# max_memory = {i:'36GiB' for i in range(6)}
# model = AutoModelForCausalLM.from_pretrained('bigscience/bloom', device_map='balanced', load_in_8bit=True,
                                            #  max_memory=max_memory)
# print(model.hf_device_map)
# print(model.get_memory_footprint() / 1024**3)

import numpy as np
import time

N = 20

a = np.random.rand(10000, 3000)
b = np.random.rand(3000, 7000)

t0 = time.time()
for i in range(N):
    foo = a@b
    max_ = np.max(foo, axis=1)
    ojnwoq = np.exp(foo)

dt = time.time() - t0
print(f'Time: {dt:.2f} s')
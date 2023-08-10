import time
import torch

import engine
from transformers import AutoModelForCausalLM

model_name = 'bigcode/starcoder'

# t0 = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_name)
# dt0 = time.time() - t0
# print(f'Default time: {dt0:.2f}s')

# del model

# t1 = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
# dt1 = time.time() - t1
# print(f'Low cpu option: {dt1:.2f}s')

# del model

# t2 = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
# dt2 = time.time() - t2
# print(f'dtype16 option: {dt2:.2f}s')

# del model

t3 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
dt3 = time.time() - t3
print(f'Low cpu option + dtype16: {dt3:.2f}s')

del model

t4 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
dt4 = time.time() - t4
print(f'Low cpu option + dtype16 + gpu: {dt4:.2f}s')


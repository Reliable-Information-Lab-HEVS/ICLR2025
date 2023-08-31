from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import time

model_name = 'facebook/opt-13b'

max_memory = {0: '10GiB', 1: '25GiB'}
device_map = 'balanced'
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map,
#                                              max_memory=max_memory)

# print(f'GPU 0: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB')
# print(f'GPU 1: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GiB')

# print(model.hf_device_map)


t0 = time.time()
config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
    device_map = infer_auto_device_map(model, max_memory=max_memory)
dt0 = time.time() - t0
print(f'Time to compute device_map: {dt0:.2f} s')

t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
dt1 = time.time() - t1
print(f'Time to load model: {dt1:.2f} s')

print(f'Total time: {dt0 + dt1:.2f} s')

print(f'GPU 0: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB')
print(f'GPU 1: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GiB')

print(model.hf_device_map)
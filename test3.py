import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

# model = engine.HFModel('bloom-560M', gpu_rank=0)
# print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
# print(model.device_map)

# prompt = "# Write a python function to multiply 2 numbers"

# t0 = time.time()
# for i in range(50):
#     out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=64,
#                 stopping_patterns=stopping.CODE_STOP_PATTERNS, gpu_rank=1)
# dt = time.time() - t0
# print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
# print(f'Time for generation: {dt:.2f} s')


max_memory = {0: '10GiB', 1:'10GiB'}
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m', device_map='auto', max_memory=max_memory)
print(model.hf_device_map)
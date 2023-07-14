import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

model = engine.HFModel('bloom-560M', gpu_rank=0)
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
print(model.device_map)

prompt = "# Write a python function to multiply 2 numbers"

t0 = time.time()
out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=200,
            stopping_patterns=stopping.CODE_STOP_PATTERNS)
dt = time.time() - t0
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')

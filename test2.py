import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

prompt = "# Write a python function to multiply 2 numbers"

model = engine.HFModel('bloom-7.1B')

for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')

t0 = time.time()
out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=200,
            stopping_patterns=stopping.CODE_STOP_PATTERNS)
dt = time.time() - t0

for i in range(torch.cuda.device_count()):
    print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')
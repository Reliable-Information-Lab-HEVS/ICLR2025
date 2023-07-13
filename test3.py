import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping

model = engine.HFModel('star-coder')
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
print(model.device_map)

prompt = "# Write a python function to multiply 2 numbers"

for i in range(50):
    out = model(prompt, max_new_tokens=512, batch_size=64, stopping_patterns=stopping.CODE_STOP_PATTERNS)
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
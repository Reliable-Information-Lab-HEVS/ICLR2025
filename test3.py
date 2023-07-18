import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

prompt = "# Write a python function to multiply 2 numbers"
max_tokens = 512
batch_size = 200

model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')
model.input_device = 0
for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')
print(model.device_map)
print(model.input_device)

input_size = model.tokenizer.encode(prompt, return_tensors='pt').shape[1]
multiplier = 2 if (model.dtype == torch.bfloat16 or model.dtype == torch.float16) else 4
inferred_mem_size = batch_size * (input_size + max_tokens) * model.tokenizer.vocab_size * multiplier / 1024**3

t0 = time.time()
out = model(prompt, max_new_tokens=max_tokens, num_return_sequences=200, batch_size=batch_size,
            stopping_patterns=None)
dt = time.time() - t0

size_out = model.tokenizer(out, return_tensors='pt', padding=True).input_ids.shape[1]
if size_out != (input_size + max_tokens):
    print(f'Early stopping of generation after {size_out} tokens total.')
    inferred_mem_size = batch_size * size_out * model.tokenizer.vocab_size * multiplier / 1024**3

for i in range(torch.cuda.device_count()):
    print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

print(f'According to calculation, memory should be {inferred_mem_size:.5f} GB')
print(f'Time for generation: {dt:.2f} s')




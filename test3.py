import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

prompt = "# Write a python function to multiply 2 numbers"

# model = engine.HFModel('codegen-16B', gpu_rank=0, device_map='auto')
# print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
# print(model.device_map)

# t0 = time.time()
# out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=100,
#             stopping_patterns=stopping.CODE_STOP_PATTERNS)
# dt = time.time() - t0
# print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
# print(f'Time for generation: {dt:.2f} s')


# model = engine.HFModel('codegen-16B', gpu_rank=0)
# print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
# print(model.device_map)
model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-16B-mono', torch_dtype=torch.float16).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-16B-mono')
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')

t0 = time.time()
# out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=100,
            # stopping_patterns=stopping.CODE_STOP_PATTERNS)
out = generation.generate_text(model, tokenizer, prompt, max_new_tokens=512, num_return_sequences=200, batch_size=100,
                               stopping_patterns=stopping.CODE_STOP_PATTERNS)
dt = time.time() - t0
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')

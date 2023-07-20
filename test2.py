import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time
import resource

# prompt = "# Write a python function to multiply 2 numbers"

# model = engine.HFModel('bloom-3B', dtype=torch.float32)

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
# else:
#     print(f'Before generation cpu: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3):.2f} GB')


# t0 = time.time()
# out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=200,
#             stopping_patterns=stopping.CODE_STOP_PATTERNS)
# dt = time.time() - t0

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
# else:
#     print(f'After generation cpu: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3):.2f} GB')
# print(f'Time for generation: {dt:.2f} s')



#
#
#

import engine
import torch

prompt = "Hello my dear"

# torch.cuda.reset_peak_memory_stats(device=0)

model = engine.HFModel('bloom-7.1B')

model_memory = torch.cuda.max_memory_allocated(0) / 1024**3

input_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda(0)
input_ids, _ = model.model._expand_inputs_for_generation(expand_size=20, input_ids=input_ids)

torch.cuda.reset_peak_memory_stats(device=0)

past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values

memory_with_grad = torch.cuda.max_memory_allocated(0) / 1024**3 - model_memory

del past_key_values
torch.cuda.reset_peak_memory_stats(device=0)

with torch.no_grad():
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values

memory_without_grad = torch.cuda.max_memory_allocated(0) / 1024**3 - model_memory

print(f'Memory with grad: {memory_with_grad:.5f} GiB')
print(f'Memory without grad: {memory_without_grad:.5f} GiB')
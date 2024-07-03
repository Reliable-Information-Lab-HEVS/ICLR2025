from TextWiz import textwiz

import torch
from transformers import AutoModelForCausalLM
import time

model = textwiz.HFCausalModel('llama3-8B-instruct')
model_ = AutoModelForCausalLM.from_pretrained(textwiz.loader.ALL_MODELS_MAPPING['llama3-8B-instruct'],
                                             torch_dtype=textwiz.loader.ALL_MODELS_DTYPES['llama3-8B-instruct'],
                                             attn_implementation='sdpa', low_cpu_mem_usage=True).cuda()
model.model = model_

prompt = 'Write an extremely long text about monkeys'

# Default mode
t0 = time.time()
foo = model(prompt, max_new_tokens=4096, min_new_tokens=4096, do_sample=False, batch_size=1,
            cache_implementation='static')
dt0 = time.time() - t0
print(f'Total time for base inference: {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

# Default compiling
t0 = time.time()
model.model = torch.compile(model.model, mode='default', fullgraph=True)
dt0 = time.time() - t0
print(f'Time for default compiling: {dt0:.2e} s')

t0 = time.time()
foo = model(prompt, max_new_tokens=4096, min_new_tokens=4096, do_sample=False, batch_size=1,
            cache_implementation='static')
dt0 = time.time() - t0
print(f'Total time for inference: with default compiling {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

model_ = AutoModelForCausalLM.from_pretrained(textwiz.loader.ALL_MODELS_MAPPING['llama3-8B-instruct'],
                                             torch_dtype=textwiz.loader.ALL_MODELS_DTYPES['llama3-8B-instruct'],
                                             attn_implementation='sdpa', low_cpu_mem_usage=True).cuda()
model.model = model_
# Better compiling
t0 = time.time()
model.model = torch.compile(model.model, mode='max-autotune', fullgraph=True)
dt0 = time.time() - t0
print(f'Time for autotune compiling: {dt0:.2e} s')

t0 = time.time()
foo = model(prompt, max_new_tokens=4096, min_new_tokens=4096, do_sample=False, batch_size=1,
            cache_implementation='static')
dt0 = time.time() - t0
print(f'Total time for inference: with autotune compiling {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

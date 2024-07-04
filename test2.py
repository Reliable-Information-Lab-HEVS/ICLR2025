from TextWiz import textwiz

import os

os.environ['TORCH_LOGS' ] = "+dynamo"
os.environ['TORCHDYNAMO_VERBOSE'] = "1"
os.environ['TRITON_PTXAS_PATH'] = "/cluster/raid/home/vacy/miniforge3/envs/llm/bin/ptxas"


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import time

model = AutoModelForCausalLM.from_pretrained(textwiz.loader.ALL_MODELS_MAPPING['llama3-8B-instruct'],
                                             torch_dtype=textwiz.loader.ALL_MODELS_DTYPES['llama3-8B-instruct'],
                                             attn_implementation='sdpa', low_cpu_mem_usage=True).cuda()
tokenizer = textwiz.load_tokenizer('llama3-8B-instruct')


prompt = 'Write an extremely long text about monkeys'
input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

# Default mode
t0 = time.time()
foo = model.generate(input_ids, max_new_tokens=4096, min_new_tokens=4096, do_sample=False,
                     cache_implementation='static', return_dict=True)
dt0 = time.time() - t0
print(f'Total time for base inference: {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

# Default compiling
t0 = time.time()
model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=True)
initial_pass = model.generate(input_ids, max_new_tokens=1, min_new_tokens=1, do_sample=False,
                              cache_implementation='static', return_dict=True)
dt0 = time.time() - t0
print(f'Time for default compiling: {dt0:.2e} s')

t0 = time.time()
foo = model.generate(input_ids, max_new_tokens=4096, min_new_tokens=4096, do_sample=False,
                     cache_implementation='static', return_dict=True)
dt0 = time.time() - t0
print(f'Total time for inference: with default compiling {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

model = AutoModelForCausalLM.from_pretrained(textwiz.loader.ALL_MODELS_MAPPING['llama3-8B-instruct'],
                                             torch_dtype=textwiz.loader.ALL_MODELS_DTYPES['llama3-8B-instruct'],
                                             attn_implementation='sdpa', low_cpu_mem_usage=True).cuda()
# Better compiling
t0 = time.time()
model.forward = torch.compile(model.forward, mode='max-autotune', fullgraph=True)
initial_pass = model.generate(input_ids, max_new_tokens=1, min_new_tokens=1, do_sample=False,
                              cache_implementation='static', return_dict=True)
dt0 = time.time() - t0
print(f'Time for autotune compiling: {dt0:.2e} s')

t0 = time.time()
foo = model.generate(input_ids, max_new_tokens=4096, min_new_tokens=4096, do_sample=False,
                     cache_implementation='static', return_dict=True)
dt0 = time.time() - t0
print(f'Total time for inference: with autotune compiling {dt0:.2e} s --- {4096 / dt0:.2f} tokens/s')

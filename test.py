import torch
import numpy as np
import argparse
import time
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain

import engine
from engine import stopping
from engine import loader, generation
from helpers import utils

# llm = agents.HuggingFaceLLM.from_name('star-coder', max_new_tokens=300)
# tools = [agents.Flake8Tool()]
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("Write code to multiply 2 numbers, then refactore it according to Flake8.")

# prompt = "Write code to multiply 2 numbers"
# t0 = time.time()
# # model = engine.HFModel('star-coder')
# model = engine.HFModel('bloom-560M')
# dt = time.time() - t0
# print(f'Time to load the model: {dt:.2f} s')

# t1 = time.time()
# res = model(prompt)
# dt1 = time.time() - t1
# print(f'Time for inference: {dt1:.2f} s')
# print(f'Output: {res}')

# model_name = 'bloom-560M'
# prompt = "Write code to multiply 2 numbers"

# model, tokenizer = loader.load_model_and_tokenizer(model_name, device_map='auto')
# print(model.hf_device_map)

# t0 = time.time()
# foo = generation.generate_text(model, tokenizer, prompt, num_return_sequences=200, batch_size=100, max_new_tokens=500)
# dt = time.time() - t0
# print(f'Time needed with auto: {dt:.2f} s')

# del model, tokenizer
# gc.collect()

# model, tokenizer = loader.load_model_and_tokenizer(model_name, device_map='sequential')
# print(model.hf_device_map)

# t1 = time.time()
# foo = generation.generate_text(model, tokenizer, prompt, num_return_sequences=200, batch_size=100, max_new_tokens=500)
# dt1 = time.time() - t1
# print(f'Time needed without auto: {dt1:.2f} s')


# del model, tokenizer
# gc.collect()

# model = AutoModelForCausalLM.from_pretrained(loader.DECODER_MODELS_MAPPING[model_name], device_map=None,
#                                                     torch_dtype='auto', load_in_8bit=False)
# model = model.to('cuda:0')
# tokenizer = loader.load_tokenizer(model_name)
# # print(model.hf_device_map)

# t2 = time.time()
# foo = generation.generate_text(model, tokenizer, prompt, num_return_sequences=200, batch_size=100, max_new_tokens=500)
# dt2 = time.time() - t2
# print(f'Time needed using cuda: {dt2:.2f} s')


# model_name = 'bloom-7.1B'
# model = AutoModelForCausalLM.from_pretrained(loader.DECODER_MODELS_MAPPING[model_name], device_map=None,
#                                                     torch_dtype=torch.float16, load_in_8bit=False).to('cuda')
# tokenizer = loader.load_tokenizer(model_name)

# prompt = "Write code to multiply 2 numbers"

# foo = generation.generate_text(model, tokenizer, prompt, num_return_sequences=1, batch_size=100, max_new_tokens=500,
#                                seed=1)
# print(f'float16: {foo}')

# del model, tokenizer


# model = AutoModelForCausalLM.from_pretrained(loader.DECODER_MODELS_MAPPING[model_name], device_map=None,
#                                                     torch_dtype='auto', load_in_8bit=False).to('cuda')
# tokenizer = loader.load_tokenizer(model_name)

# prompt = "Write code to multiply 2 numbers"

# foo2 = generation.generate_text(model, tokenizer, prompt, num_return_sequences=1, batch_size=100, max_new_tokens=500,
#                                seed=1)
# print(f'original: {foo2}')

# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))

prompt = "# Write a python function to multiply 2 numbers"
batch_size = 200
max_tokens = 200

model1 = engine.HFModel('bloom-560M', gpu_rank=0)
for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
t0 = time.time()
out = model1(prompt, max_new_tokens=max_tokens, num_return_sequences=200, batch_size=batch_size,
             stopping_patterns=None)
dt = time.time() - t0
for i in range(torch.cuda.device_count()):
    print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')

torch.cuda.reset_peak_memory_stats(device=0)
torch.cuda.reset_peak_memory_stats(device=1)

model2 = engine.HFModel('bloom-560M', device_map='auto')
for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
t0 = time.time()
out = model1(prompt, max_new_tokens=max_tokens, num_return_sequences=200, batch_size=batch_size,
             stopping_patterns=None)
dt = time.time() - t0
for i in range(torch.cuda.device_count()):
    print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')
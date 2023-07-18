import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time

# prompt = "# Write a python function to multiply 2 numbers"
# Prompt of about 500 tokens. Courtesy of ChatGPT for making-up random stuff of the appropriate length.
prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations. Belonging to the primate order, monkeys are found in various parts of the world, inhabiting diverse habitats ranging from tropical rainforests to savannas. With their intelligence, agility, and unique characteristics, monkeys have left an indelible mark on our collective imagination.
"""
max_tokens = 512
batch_size = 200

model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')
# model.input_device = 0
print(model.device_map)
print(model.input_device)
input_size = model.tokenizer.encode(prompt, return_tensors='pt').shape[1]
print(f'Input sequence size: {input_size}')

for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

# multiplier = 2 if (model.dtype == torch.bfloat16 or model.dtype == torch.float16) else 4
# inferred_mem_size = batch_size * (input_size + max_tokens) * model.tokenizer.vocab_size * multiplier / 1024**3

# t0 = time.time()
# out = model(prompt, max_new_tokens=max_tokens, num_return_sequences=200, batch_size=batch_size,
#             stopping_patterns=None)
# dt = time.time() - t0

# size_out = model.tokenizer(out, return_tensors='pt', padding=True).input_ids.shape[1]
# if size_out != (input_size + max_tokens):
#     print(f'Early stopping of generation after {size_out} tokens total.')
#     inferred_mem_size = batch_size * size_out * model.tokenizer.vocab_size * multiplier / 1024**3

# for i in range(torch.cuda.device_count()):
#     print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

# print(f'According to calculation, memory should be {inferred_mem_size:.5f} GB')
# print(f'Time for generation: {dt:.2f} s')

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

input_ids = model.tokenizer.encode(prompt, return_tensors='pt')
large_input, _ = model.model._expand_inputs_for_generation(expand_size=200, input_ids=input_ids)

out = model.model(large_input).logits

for i in range(torch.cuda.device_count()):
    print(f'After model forward gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

print(f'dtype: {out.dtype}')
print(f'shape: {out.shape}')
print(f'memory: {out.element_size() * out.nelement() / 1024**3}')

del out
gc.collect()

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

out2 = model(prompt, num_return_sequences=200, max_new_tokens=1)

for i in range(torch.cuda.device_count()):
    print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')


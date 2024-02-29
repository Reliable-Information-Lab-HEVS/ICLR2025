
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from TextWiz import textwiz
from TextWiz.memory_estimator import LARGE_TEXT

model = textwiz.HFModel('zephyr-7B-beta')
# new_model = AutoModelForCausalLM.from_pretrained('HuggingFaceH4/zephyr-7b-beta', torch_dtype=torch.bfloat16,
#                                              low_cpu_mem_usage=True, attn_implementation='sdpa').cuda()
# model = model.to_bettertransformer()
# model.model = new_model
model('Hello please do your magic', num_return_sequences=1, batch_size=1, max_new_tokens=2)

large_tokens = model.tokenizer.encode(LARGE_TEXT)
N = 10

prompt = model.tokenizer.decode(large_tokens[:4000], skip_special_tokens=True)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
    with torch.no_grad():
        prompt_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda()

        torch.cuda.reset_peak_memory_stats()
        actual_peak = torch.cuda.max_memory_allocated() / 1024**3

        # Single forward pass, caching past key values
        output = model.model(prompt_ids, use_cache=True)

        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak
        print(f'Memory first forward: {memory_used}')
            
        past_key_values_memory = output.past_key_values

        next_token_logits = output.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([prompt_ids, next_tokens[:, None]], dim=-1)

        # Single forward pass, with past key values
        torch.cuda.reset_peak_memory_stats()
        actual_peak = torch.cuda.max_memory_allocated() / 1024**3

        output = model.model(input_ids, use_cache=True, past_key_values=output.past_key_values)

        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak
        print(f'Memory second forward: {memory_used}')
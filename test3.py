
import time
import numpy as np
import torch
import gc
from transformers import AutoModelForCausalLM

from TextWiz import textwiz
from TextWiz.memory_estimator import LARGE_TEXT

def memory_usage(past_key_values):
    """Recursively compute the memory footprint of past key values (in bytes).
    """

    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.nelement() * past_key_values.element_size()
    elif isinstance(past_key_values[0], torch.Tensor):
        return sum([x.nelement() * x.element_size() for x in past_key_values])
    else:
        return sum([memory_usage(x) for x in past_key_values])
    

model = textwiz.HFModel('zephyr-7B-beta')
# new_model = AutoModelForCausalLM.from_pretrained('HuggingFaceH4/zephyr-7b-beta', torch_dtype=torch.bfloat16,
#                                              low_cpu_mem_usage=True, attn_implementation='sdpa').cuda()
# model = model.to_bettertransformer()
# model.model = new_model
model('Hello please do your magic', num_return_sequences=1, batch_size=1, max_new_tokens=2)

large_tokens = model.tokenizer.encode(LARGE_TEXT)
N = 10

prompt = model.tokenizer.decode(large_tokens[:4000], skip_special_tokens=True)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    with torch.no_grad():
        prompt_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda()

        torch.cuda.reset_peak_memory_stats()
        actual_peak = torch.cuda.max_memory_allocated() / 1024**3

        # Single forward pass, caching past key values
        output = model.model.generate(prompt_ids, use_cache=True, return_dict_in_generate=True, pad_token_id=model.tokenizer.eos_token_id,
                                           max_new_tokens=1, do_sample=False)

        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak
        print(f'Memory first forward: {memory_used}')
            
        past_key_values = output.past_key_values
        past_key_values_memory = memory_usage(past_key_values) / 1024**3
        print(f'Past key values: {past_key_values_memory}')

        input_ids = torch.cat([prompt_ids, torch.tensor([[47]])], dim=-1)

        del output
        gc.collect()

        # Single forward pass, with past key values
        torch.cuda.reset_peak_memory_stats()
        actual_peak = torch.cuda.max_memory_allocated() / 1024**3

        output = model.model.generate(input_ids, use_cache=True, past_key_values=past_key_values, pad_token_id=model.tokenizer.eos_token_id,
                                           max_new_tokens=1, do_sample=False)

        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak
        print(f'Memory second forward: {memory_used}')

import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from TextWiz import textwiz
from TextWiz.memory_estimator import LARGE_TEXT

model = textwiz.HFModel('llama2-7B-chat')
# new_model = AutoModelForCausalLM.from_pretrained('HuggingFaceH4/zephyr-7b-beta', torch_dtype=torch.bfloat16,
#                                              low_cpu_mem_usage=True, attn_implementation='sdpa').cuda()
# model = model.to_bettertransformer()
# model.model = new_model
model('Hello please do your magic', num_return_sequences=1, batch_size=1, max_new_tokens=2)

large_tokens = model.tokenizer.encode(LARGE_TEXT)
sizes = [10, 100, 1000, 2000, 4000]
N = 10

times = {}
memories = {}

for input_size in sizes:
    prompt = model.tokenizer.decode(large_tokens[:input_size], skip_special_tokens=True)
    gen_times = []
    gen_mem = []

    for i in range(N):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        actual_peak = torch.cuda.max_memory_allocated() / 1024**3

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            foo = model(prompt, num_return_sequences=1, batch_size=1, max_new_tokens=2)

        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak

        gen_times.append(time.time() - t0)
        gen_mem.append(memory_used)

    times[input_size] = np.mean(gen_times)
    memories[input_size] = np.mean(gen_mem)

print(f'time: {times}')
print(f'memory: {memories}')

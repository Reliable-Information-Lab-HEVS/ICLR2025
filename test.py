
import time
import numpy as np
import torch

from TextWiz import textwiz
from TextWiz.memory_estimator import LARGE_TEXT

model = textwiz.HFModel('zephyr-7B-beta')
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
        actual_peak = torch.cuda.memory_allocated() / 1024**3

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            foo = model(prompt, num_return_sequences=1, batch_size=1, max_new_tokens=2)

        gen_times.append(time.time() - t0)
        # memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak

    times[input_size] = np.mean(gen_times)
    memories[input_size] = np.mean(gen_mem)

print(f'time: {times}')
print(f'memory: {memories}')

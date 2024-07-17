from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_name = 'meta-llama/Meta-Llama-3-8B'
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='flash_attention_2',
                                            torch_dtype=dtype, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Random token sequence
input = torch.randint(0, 1000, (1, 500), device='cuda')

times = []
for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Generate 200 tokens
    out = model.generate(input, max_new_tokens=200, min_new_tokens=200, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)  # time in seconds


print(np.mean(times))
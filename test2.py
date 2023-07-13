import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from engine import generation
from engine import stopping

model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', torch_dtype=torch.bfloat16).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder')
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')

prompt = "# Write a python function to multiply 2 numbers"

for i in range(50):
    out = generation.generate_text(model, tokenizer, prompt, max_new_tokens=512, num_return_sequences=200, batch_size=64,
                                   stopping_patterns=stopping.CODE_STOP_PATTERNS)
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')


# gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
# print(f'Full gpu mem: {gpu_memory:.2f} G')
# # Say we only have access to a portion of that memory for our model
# gpu_memory = 0.8 * gpu_memory
# print(f'0.8 gpu mem: {gpu_memory:.2f} G')

# model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', torch_dtype=torch.bfloat16).to('cuda:0')
# reserved_mem = torch.cuda.memory_reserved() / 1024**3
# print(f'Reserved mem:{reserved_mem:.2f} G')
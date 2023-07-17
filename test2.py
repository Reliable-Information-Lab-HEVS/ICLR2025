import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from engine import generation
from engine import stopping
import time

model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', torch_dtype=torch.bfloat16).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder')
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')

prompt = "# Write a python function to multiply 2 numbers"

t0 = time.time()
for i in range(50):
    out = generation.generate_text(model, tokenizer, prompt, max_new_tokens=512, num_return_sequences=200, batch_size=200,
                                   stopping_patterns=stopping.CODE_STOP_PATTERNS)
dt = time.time() - t0
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
print(f'Time for generation: {dt:.2f} s')
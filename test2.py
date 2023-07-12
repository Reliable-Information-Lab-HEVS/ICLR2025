import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from engine import generation

model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', torch_dtype=torch.bfloat16).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder')
print(f'Before generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')

prompt = "# Write a python function to multiply 2 numbers"

out = generation.generate_text(model, tokenizer, prompt, max_new_tokens=512, batch_size=64)
print(f'After generation: {(torch.cuda.max_memory_allocated(0) / 1024**3):.2f} GB')
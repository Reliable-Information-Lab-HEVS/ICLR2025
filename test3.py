import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from engine import generation

model1 = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b', torch_dtype=torch.float16, device_map='balanced_low_0')
print(model1.hf_device_map)
# tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')

del model1
gc.collect()

max_memory = {0: '6GiB', 1:'6GiB', 2:'6GiB', 'cpu':'0GiB'}
model2 = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b', torch_dtype=torch.float16, device_map='balanced_low_0',
                                              max_memory=max_memory)
print(model2.hf_device_map)
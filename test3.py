from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'facebook/opt-13b'

max_memory = {0: '10GiB', 1: '25GiB'}
device_map = 'balanced_low_0'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map,
                                             max_memory=max_memory)

print(f'GPU 0: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB')
print(f'GPU 1: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GiB')

print(model.hf_device_map)







import torch
import numpy as np
import argparse
import time

import loader
import engine
import utils

model = 'bloom'

t0 = time.time()
model, tokenizer = loader.load_model_and_tokenizer(model, quantization=True)
dt = time.time() - t0
print(f'Time to load bloom quantized: {dt:.2f} s')

prompt = 'Write Python code to create a model with Pytorch.'
t1 = time.time()
out = engine.generate_text(model, tokenizer, prompt, max_new_tokens=150, num_return_sequences=5)
dt1 = time.time() - t1
print(f'Time for inference: {dt1:.2f} s')
print(utils.format_output(out))

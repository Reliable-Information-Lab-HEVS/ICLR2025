import torch
import numpy as np
import argparse
import time

import loader
import utils

parser = argparse.ArgumentParser(description='Clustering of the memes')
parser.add_argument('--model_name', type=str, default='bloom-560M', choices=loader.AUTHORIZED_MODELS,
                    help='The model to use.')
args = parser.parse_args()
model_name = args.model_name

model, tokenizer = loader.load_model_and_tokenizer(model_name)
print(model.device)
print(model.hf_device_map)

prompt = 'Hello, my dog is cute'
t0 = time.time()
predictions = loader.generate_text(model, tokenizer, prompt)
dt = time.time() - t0
print(f'Time for prediction: {dt:.2f} s')
print(utils.format_output(predictions))

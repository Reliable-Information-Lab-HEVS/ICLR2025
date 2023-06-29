import torch
import numpy as np
import argparse
import time

import loader
import engine
import utils
from huggingface_hub import login
import os

model = 'star-coder'

token_file = os.path.join(utils.ROOT_FOLDER, '.hf_token.txt')
with open(token_file, 'r') as file:
    # Read lines and remove whitespaces
    token = file.readline().strip()

login(token)

model, tokenizer = loader.load_model_and_tokenizer(model, quantization=False)
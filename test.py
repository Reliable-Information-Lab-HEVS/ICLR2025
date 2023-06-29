import torch
import numpy as np
import argparse
import time

import loader
import engine
import utils

model = 'star-coder'

model, tokenizer = loader.load_model_and_tokenizer(model, quantization=False)
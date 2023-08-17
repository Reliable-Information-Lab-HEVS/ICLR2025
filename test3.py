import time
import torch

import engine
from helpers import datasets
from transformers import AutoModelForCausalLM

model_name = 'star-chat-alpha'

model = engine.HFModel(model_name)
prompt = datasets.HumanEval()[0]['prompt']
out = model(prompt, truncate_prompt_from_output=True)
print(out)



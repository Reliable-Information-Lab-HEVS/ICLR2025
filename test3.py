import torch
import time

import engine
from engine import loader, stopping
from helpers import utils, datasets

HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 256,
    'min_new_tokens': 0,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 1,
    'stopping_patterns': None,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

model_name = 'llama2-7B-chat'

model = engine.HFModel(model_name)
prompt = 'Hello there, could you tell me what is the meaning of life?'

t0 = time.time()
completions = model(prompt, temperature=0.8, **HUMAN_EVAL_GENERATION_KWARGS)
dt = time.time() - t0
print(f'It took {dt:.2f} s')

print(completions)
import torch
import time

import engine
from engine import loader, stopping
from helpers import utils, datasets

HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'stopping_patterns': stopping.StoppingType.PYTHON_HUMAN_EVAL,
    'prompt_template_mode': 'generation',
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

model_name = 'llama2-7B'

model = engine.HFModel(model_name)
data = datasets.HumanEval()
prompt = data[0]['prompt']
stop = stopping.StoppingType.PYTHON_HUMAN_EVAL

t0 = time.time()
completions = model(prompt, temperature=0.8, **HUMAN_EVAL_GENERATION_KWARGS)
dt = time.time() - t0
print(f'It took {dt:.2f} s')
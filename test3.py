import time
import torch

import engine
from helpers import datasets
from transformers import AutoModelForCausalLM

from engine import loader
from helpers import datasets

dataset = datasets.HumanEvalInstruct()
sample = dataset[0]
prompt = sample['instruction'] + sample['context']

HUMAN_EVAL_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'top_k': 0,
    'top_p': 1.,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
    'stopping_patterns': False
}

model = loader.load_model('vicuna-7B')
out = model(prompt, **HUMAN_EVAL_GREEDY_GENERATION_KWARGS)

print(prompt + out)



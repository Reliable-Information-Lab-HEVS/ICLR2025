import time
import torch

import engine
from helpers import datasets
from transformers import AutoModelForCausalLM

import engine
from helpers import datasets

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

# dataset = datasets.HumanEvalInstruct()
# sample = dataset[0]
# prompt = sample['instruction'] + sample['context']

# model = engine.HFModel('vicuna-7B')
# out = model(prompt, batch_size=1, **HUMAN_EVAL_GREEDY_GENERATION_KWARGS)
# print(prompt + out)


dataset = datasets.HumanEval()
sample = dataset[17]
prompt = sample['prompt']

model = engine.HFModel('star-chat-beta')
out = model(prompt, batch_size=1, **HUMAN_EVAL_GREEDY_GENERATION_KWARGS)
print(prompt + out)


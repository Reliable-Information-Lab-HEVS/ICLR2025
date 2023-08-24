import engine
from engine import stopping
from helpers import datasets

import human_eval

model = engine.HFModel('vicuna-7B')

HUMAN_EVAL_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'top_k': 0,
    'top_p': 1.,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

dataset = datasets.HumanEval()
sample = dataset[26]

prompt = sample['prompt']

completions = [model(prompt, temperature=1., batch_size=1, stopping_patterns=None,
                    **HUMAN_EVAL_GREEDY_GENERATION_KWARGS)]
print(completions)

true_completions = human_eval.extract_completions(completions, sample)
results = [{'task_id': 26, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]

print('\n\n')
print(true_completions)





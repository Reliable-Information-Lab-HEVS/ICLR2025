import os
import copy

from helpers import utils, datasets
from engine import stopping


path = os.path.join(utils.RESULTS_FOLDER, 'HumanEval_completions')
models = [os.path.join(path, dir) for dir in os.listdir(path) if not dir.startswith('.')]
files_in = [os.path.join(model, f'temperature_0.0.jsonl') for model in models]
files_out = [os.path.join(model, f'temperature_0.0_modified.jsonl') for model in models]

problems = datasets.HumanEval().samples_by_id()

for file_in, file_out in zip(files_in, files_out):
    completions = utils.load_jsonl(file_in)
    new_completions = []
    for completion in completions:
        new_completion = copy.deepcopy(completion)
        new_completion['completion'] = stopping.post_process_sequences([completion['completion']])[0]
        new_completions.append(new_completion)

    utils.save_jsonl(new_completions, file_out)


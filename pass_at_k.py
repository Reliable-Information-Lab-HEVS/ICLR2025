import os
from tqdm import tqdm

from code_execution import evaluation
from helpers import utils

path = os.path.join(utils.RESULTS_FOLDER, 'HumanEval_completions_new_results')
model_directories = [os.path.join(path, dir) for dir in os.listdir(path) if not dir.startswith('.')]
files = [os.path.join(dir, f'temperature_0.0_modified.jsonl') for dir in model_directories]
# temperatures = (0.,)


passes = {}
for file in tqdm(files):
    _, model, _ = file.rsplit('/', 2)
    pass_at_k = evaluation.evaluate_pass_at_k(file)
    passes[model] = pass_at_k

print(passes)
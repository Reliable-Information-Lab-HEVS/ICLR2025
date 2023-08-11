import os
from tqdm import tqdm

from code_execution import evaluation
from helpers import utils

path = os.path.join(utils.RESULTS_FOLDER, 'HumanEval_completions')
model_directories = [os.path.join(path, dir) for dir in os.listdir(path) if not dir.startswith('.')]
# files = [os.path.join(dir, f'temperature_0.0_modified.jsonl') for dir in model_directories]
files = []
for model_dir in model_directories:
    files.extend([os.path.join(model_dir, file) for file in os.listdir(model_dir) if 'modified' in file])

if __name__ == '__main__':
    for file in tqdm(files):
        _ = evaluation.evaluate_functional_correctness(file, n_workers=6, timeout=3)

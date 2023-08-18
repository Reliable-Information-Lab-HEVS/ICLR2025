import os
from tqdm import tqdm
import numpy as np

from engine import loader
from code_execution import evaluation
from helpers import utils

path = os.path.join(utils.RESULTS_FOLDER, 'HumanEval_completions_results')


model_directories = [os.path.join(path, dir) for dir in os.listdir(path) if not dir.startswith('.')]
files = []
for model_dir in model_directories:
    # files.extend([os.path.join(model_dir, file) for file in os.listdir(model_dir) if 'modified' in file])
    files.extend([os.path.join(model_dir, file) for file in os.listdir(model_dir)])


passes = []
for file in tqdm(files):
    _, model, filename = file.rsplit('/', 2)
    model_size = loader.ALL_MODELS_PARAMS_MAPPING[model]
    if 'infill' in filename:
        model += '_infill'
    pass_at_k = evaluation.evaluate_pass_at_k(file)
    passes.append({'model': model, 'pass@1': pass_at_k['pass@1'], 'model_size': model_size})

# Sort according to name then model size
passes = sorted(passes, key=lambda x: (x['model'].split('-')[0], x['model_size']))
names = [x['model'] for x in passes]
pass_at_1 = np.array([x['pass@1'] for x in passes])


import matplotlib.pyplot as plt
plt.figure()
plt.barh(np.arange(len(pass_at_1)), width=pass_at_1)
yticks, _ = plt.yticks()
plt.yticks(yticks, names, rotation='horizontal')
plt.xlabel('pass@1 (greedy decoding)')
plt.ylabel('Model')
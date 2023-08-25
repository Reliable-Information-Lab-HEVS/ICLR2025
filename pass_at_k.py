import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from engine import loader
from code_execution import evaluation
from helpers import utils

# Files are at: utils.RESULTS_FOLDER/HumanEval.../results/model/temperature.jsonl

human_eval_folders = [os.path.join(utils.RESULTS_FOLDER, folder) for folder in os.listdir(utils.RESULTS_FOLDER) if
                      (not folder.startswith('.') and 'HumanEval' in folder)]
# Only keep those that were not already processed
human_eval_folders = [folder for folder in human_eval_folders if 'results' in os.listdir(folder)]

# This contains utils.RESULTS_FOLDER/HumanEval.../completions
human_eval_results = [os.path.join(folder, 'results') for folder in human_eval_folders]

files = []
for benchmark in human_eval_results:
    for model_dir in os.listdir(benchmark):
        if not model_dir.startswith('.'):
            full_path = os.path.join(benchmark, model_dir)
            files.extend([os.path.join(full_path, file) for file in os.listdir(full_path) if not file.startswith('.')])


passes = []
for file in tqdm(files):
    _, benchmark, _, model, filename = file.rsplit('/', 4)
    model_size = loader.ALL_MODELS_PARAMS_MAPPING[model]
    pass_at_k = evaluation.evaluate_pass_at_k(file)
    passes.append({'model': model, 'pass@1': pass_at_k['pass@1'], 'model_size': model_size, 'benchmark': benchmark,
                   'model_family': model.rsplit('-', 1)[0]})

benchmarks = set([x['benchmark'] for x in passes])
models = set([x['model'] for x in passes])

dics = []
for model in models:
    dic = {}
    dic['model'] = model
    model_results = [x for x in passes if x['model'] == model]
    dic['model_family'] = model_results[0]['model_family']
    dic['model_size'] = model_results[0]['model_size']
    for res in model_results:
        benchmark = res['benchmark']
        dic[f'{benchmark}_pass@1'] = res['pass@1']*100
    dics.append(dic)

benchmark_df = pd.DataFrame(dics).set_index('model').sort_values(['model_family', 'model_size']).drop(columns=['model_family', 'model_size']).fillna('-')

dfs = []
for benchmark in benchmarks:
    df = pd.DataFrame([x for x in passes if x['benchmark'] == benchmark]).set_index('model').sort_values(['model_family', 'model_size'])
    dfs.append(df)

# # Sort according to name then model size
# passes = sorted(passes, key=lambda x: (x['model'].split('-')[0], x['model_size']))
# names = [x['model'] for x in passes]
# pass_at_1 = np.array([x['pass@1'] for x in passes])


# import matplotlib.pyplot as plt
# plt.figure()
# plt.barh(np.arange(len(pass_at_1)), width=pass_at_1)
# yticks, _ = plt.yticks()
# plt.yticks(yticks, names, rotation='horizontal')
# plt.xlabel('pass@1 (greedy decoding)')
# plt.ylabel('Model')
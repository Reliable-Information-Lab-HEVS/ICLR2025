import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from engine import loader
from code_execution import evaluation
from helpers import utils
from helpers import process

def model_wise_pass_at_k(to_df: bool = True, NaN: str | None = None):

    files = process.extract_all_human_eval_filenames(category='results')

    # Compute the pass at k for every result file
    passes = []
    for file in files:
        attributes = process.parse_human_eval_filename(file)
        benchmark = attributes['benchmark_name']
        dtype = attributes['dtype']
        if dtype == 'int4' or dtype == 'int8':
            full_name = benchmark + '_' + dtype
        else:
            full_name = benchmark
        model = attributes['model']
        model_size = loader.ALL_MODELS_PARAMS[model]
        model_family = loader.ALL_MODELS_FAMILY[model]

        pass_at_k = evaluation.evaluate_pass_at_k(file)
        passes.append({'model': model, 'pass@1': pass_at_k['pass@1'], 'model_size': model_size, 'benchmark': full_name,
                    'model_family': model_family})
        
    # Swap to a "by model view"
    models = set([x['model'] for x in passes])

    dics = []
    for model in models:
        dic = {}
        dic['model'] = model

        # Find all results corresponding to current model
        model_results = [x for x in passes if x['model'] == model]
        dic['model_family'] = model_results[0]['model_family']
        dic['model_size'] = model_results[0]['model_size']
        for res in model_results:
            benchmark = res['benchmark']
            dic[f'{benchmark}_pass@1'] = res['pass@1']*100

        dics.append(dic)

    if to_df:
        df = pd.DataFrame(dics).set_index('model').sort_values(['model_family', 'model_size'])
        df = df.drop(columns=['model_family', 'model_size'])
        if NaN is not None:
            df = df.fillna(NaN)
        return df
    else:
        return dics
    


# dfs = []
# for benchmark in benchmarks:
#     df = pd.DataFrame([x for x in passes if x['benchmark'] == benchmark]).set_index('model').sort_values(['model_family', 'model_size'])
#     dfs.append(df)




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
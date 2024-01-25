import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import utils
from TextWiz.textwiz import loader

DATASETS = ['AATK', 'AATK_english', 'AATK_english_v2']
CATEGORIES = ['completions', 'results']

def get_folder(dataset: str, model_name: str, dtype_category: str) -> str:
    """Return the folder upon which to save the results of the given AATK benchmark.

    Parameters
    ----------
    dataset : str
        The dataset used.
    model_name : str
        The model name.
    dtype_category : str
        Dtype used by the model.

    Returns
    -------
    str
        Folder where we save the benchmark results.
    """

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')
    
    path = os.path.join(utils.RESULTS_FOLDER , dataset, 'completions', model_name, dtype_category)
        
    return path


def parse_filename(filename: str) -> dict:
    """Parse a filename corresponding to a AATK result file, and return the attributes that were used
    to generate it.

    Parameters
    ----------
    filename : str
        The filename.

    Returns
    -------
    dict
        A dict corresponding to the attributes.
    """

    # The file format is: utils.RESULTS_FOLDER/dataset/category/model/dtype/temperature.jsonl

    _, dataset, category, model, dtype, temperature_name = filename.rsplit('/', 5)

    associated_results_folder = os.path.join(utils.RESULTS_FOLDER, dataset, 'results')
    associated_results_file = os.path.join(associated_results_folder, model, dtype, temperature_name)
    associated_completions_file = os.path.join(utils.RESULTS_FOLDER, dataset, 'completions', model, dtype,
                                               temperature_name)
    
    temperature = float(temperature_name.split('_', 1)[1].rsplit('.', 1)[0])

    out = {
        'category': category,
        'model': model,
        'dtype': dtype,
        'temperature': temperature,
        'dataset': dataset,
        'associated_results_folder': associated_results_folder,
        'associated_results_file': associated_results_file,
        'associated_completions_file': associated_completions_file,
        }
    
    return out


def extract_filenames(dataset: str = 'AATK', category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to the AATK benchmark.

    Parameters
    ----------
    dataset : str
        The name of the dataset, by default `AATK`.
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.
    only_unprocessed : bool, optional
        If `True` and `category='completions'`, will keep only files from benchmarks for which the 'results' 
        folder does not exist. By default True.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    # The file format is: utils.RESULTS_FOLDER/dataset/category/model/dtype/temperature.jsonl

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, dataset, category)

    if not os.path.isdir(benchmark_path):
        raise RuntimeError('The path to the current benchmark does not exist.')
    
    files = []
    for model_dir in os.listdir(benchmark_path):
        if not model_dir.startswith('.'):
            model_path = os.path.join(benchmark_path, model_dir)
            for dtype_dir in os.listdir(model_path):
                if not dtype_dir.startswith('.'):
                    full_path = os.path.join(model_path, dtype_dir)
                    existing_files = [os.path.join(full_path, file) for file in os.listdir(full_path) \
                                      if not file.startswith('.')]
                    
                    if category == 'completions' and only_unprocessed:
                        # Add it only if corresponding results file does not already exist
                        for file in existing_files:
                            if not os.path.exists(file.replace('completions', 'results')):
                                files.append(file)
                    else:
                        files.extend(existing_files)
                    
    return files


def extract_all_filenames(category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to all benchmarks of AATK.

    Parameters
    ----------
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.
    only_unprocessed : bool, optional
        If `True` and `category='completions'`, will keep only files from benchmarks for which the 'results' 
        folder does not exist. By default True.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')
    

    files = []
    for dataset in DATASETS:

        try:
            existing_files = extract_filenames(dataset, category=category, only_unprocessed=only_unprocessed)
        except RuntimeError:
            continue

        files.extend(existing_files)

    return files


def model_wise_results(dataset: str = 'AATK'):

    files = extract_filenames(dataset, category='results')

    out = []
    for file in files:
        res = utils.load_jsonl(file)
        model = parse_filename(file)['model']

        model_size = loader.ALL_MODELS_PARAMS[model]
        model_family = loader.ALL_MODELS_FAMILY[model]

        tot_valid = sum([x['valid'] for x in res])
        tot_vulnerable = sum([x['vulnerable'] for x in res if x['vulnerable'] is not None])

        # fraction of valid files that are vulnerable
        frac_vulnerable = tot_vulnerable / tot_valid * 100
        # fraction of files that are valid
        frac_valid = tot_valid / (25 * len(res)) * 100

        out.append({'model': model, 'model_size': model_size, 'model_family': model_family,
                    'valid': frac_valid, 'vulnerable': frac_vulnerable})
        
    out = sorted(out, key=lambda x: (x['model_family'], x['model_size'], x['model']))

    df = pd.DataFrame.from_records(out)
    df = df.drop(columns=['model_family', 'model_size'])
    df.set_index('model', inplace=True)

    return df


def scenario_stats(filename: str):

    result = utils.load_jsonl(filename)

    model = parse_filename(filename)['model']

    vulnerable_by_id = defaultdict(list)
    valid_by_id = defaultdict(list)
    for res in result:
        vulnerable_by_id[res['id']].append(res['vulnerable'])
        valid_by_id[res['id']].append(res['valid'])

    weighted_mean, weighted_std, frac_vulnerable, frac_valid, x, y = {}, {}, {}, {}, {}, {}
    for key in valid_by_id.keys():
        weighted_mean[key] = np.average(vulnerable_by_id[key], weights=valid_by_id[key])
        weighted_std[key] = np.sqrt(np.average((vulnerable_by_id[key]-weighted_mean[key])**2, weights=valid_by_id[key]))

        frac_vulnerable[key] = np.sum(vulnerable_by_id[key]) / np.sum(valid_by_id[key]) * 100
        frac_valid[key] = np.sum(valid_by_id[key]) / (25 * len(valid_by_id[key])) * 100

        x[key] = []
        for a, b in zip(vulnerable_by_id[key], valid_by_id[key]):
            x[key].append(a/b if b != 0 else 0)
        x[key] = np.array(x[key])
        y[key] = np.array(valid_by_id[key]) / 25

    # means = {key: np.mean(value) for key, value in res_by_id.items()}
    # stds = {key: np.std(value) for key, value in res_by_id.items()}
    # vals = {key: f'${means[key]:.1f} \pm {stds[key]:.1f}$' for key in means.keys()}

    vals = {key: f'${weighted_mean[key]:.1f} \pm {weighted_std[key]:.1f}$' for key in weighted_std.keys()}
    # vals = {key: f'${frac_valid[key]:.1f} \pm {frac_vulnerable[key]:.1f}$' for key in weighted_std.keys()}

    # return vals, model
    return x, y, model


def scenarios_wise_stats():

    files = extract_filenames('AATK_english', category='results')

    results = [scenario_stats(file) for file in files]

    assert all(sorted(x[0].keys()) == sorted(results[0][0].keys()) for x in results)
    
    df = pd.DataFrame(index=results[0][0].keys())
    for values, name in results:
        df[name] = df.index.map(values)

    return df




def paper_boxplot(filename: str | None = None):

    # files = [
    # '/Users/cyrilvallez/Desktop/LLMs/results/AATK_perplexity/results/code-llama-34B-instruct/bfloat16/temperature_0.2.jsonl',
    # '/Users/cyrilvallez/Desktop/LLMs/results/AATK_perplexity/results/llama2-70B-chat/float16/temperature_0.2.jsonl',
    # '/Users/cyrilvallez/Desktop/LLMs/results/AATK_perplexity/results/star-chat-alpha/float16/temperature_0.2.jsonl'
    # ]

    files = extract_filenames('AATK_english', category='results')

    name_mapping = {'code-llama-34B-instruct': 'CodeLlama 34B - Instruct', 'llama2-70B-chat': 'Llama2 70B - Chat',
                    'star-chat-alpha': 'StarChat (alpha)'}
    model_names = ['star-chat-alpha', 'code-llama-34B-instruct', 'llama2-70B-chat']

    Py_per_model = {}

    for file in files:
        result = utils.load_jsonl(file)
        model = parse_filename(file)['model']
        Py = defaultdict(list)

        for res in result:
            id = res['id']
            Py[id].append(0 if res['valid'] == 0 else res['vulnerable'] / res['valid'])

        Py_per_model[model] = Py

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(4.8*2, 6.4/1.2))


    ticks = list(Py_per_model['code-llama-34B-instruct'].keys())
    new_ticks = [x.rsplit('-', 1)[0] + ' - ' + x.rsplit('-', 1)[1] for x in ticks]
    ticks_mapping = {tick: new_tick for tick, new_tick in zip(ticks, new_ticks)}

    tot = 0
    for ax, model in zip(axes, model_names):
        tot += 1
        df = pd.DataFrame(Py_per_model[model])
        df.rename(columns=ticks_mapping, inplace=True)
        ax.set_axisbelow(True)
        ax.grid(axis='x')
        sns.boxplot(df, orient='h', ax=ax)
        ax.set(xlabel=f'Distribution of $P_y$')
        if tot == 1:
            ax.set(ylabel='Prompt')
        ax.set(title=name_mapping[model])

    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
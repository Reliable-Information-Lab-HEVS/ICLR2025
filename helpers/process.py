"""This file provides helper functions to work with the HumanEval results.
"""

import os

import pandas as pd

from engine import loader
from helpers import utils

CATEGORIES = ['completions', 'results']

def parse_human_eval_filename(filename: str) -> dict:
    """Parse a filename corresponding to a human eval result file, and return the attributes that were used
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

    # The file format is: utils.RESULTS_FOLDER/benchmark/category/model/dtype/temperature.jsonl

    _, benchmark_name, category, model, dtype, temperature_name = filename.rsplit('/', 5)
    temperature = float(temperature_name.split('_', 1)[1].rsplit('.', 1)[0])
    dataset, mode = benchmark_name.split('_', 1)
    if dataset == 'HumanEvalInstruct':
        mode, use_context = mode.split('_', 1)
    else:
        use_context = None

    associated_results_folder = os.path.join(utils.RESULTS_FOLDER, benchmark_name, 'results')
    associated_results_file = os.path.join(associated_results_folder, model, dtype, temperature_name)
    associated_completions_file = os.path.join(utils.RESULTS_FOLDER, benchmark_name, 'completions', model, dtype,
                                               temperature_name)

    out = {
        'benchmark_name': benchmark_name,
        'category': category,
        'model': model,
        'dtype': dtype,
        'temperature': temperature,
        'dataset': dataset,
        'mode': mode,
        'use_context': use_context,
        'associated_results_folder': associated_results_folder,
        'associated_results_file': associated_results_file,
        'associated_completions_file': associated_completions_file,
        }
    
    return out



def extract_human_eval_filenames(benchmark: str, category: str = 'completions') -> list[str]:
    """Return all filenames corresponding to a HumanEval `benchmark`.

    Parameters
    ----------
    benchmark : str
        The name of the benchmark, i.e. `HumanEval_default`, or `HumanEval_generation`.
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    # The file format is: utils.RESULTS_FOLDER/benchmark/category/model/dtype/temperature.jsonl

    assert 'HumanEval' in benchmark, 'The benchmark name is not correct'

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, benchmark, category)
    
    files = []
    for model_dir in os.listdir(benchmark_path):
        if not model_dir.startswith('.'):
            model_path = os.path.join(benchmark_path, model_dir)
            for dtype_dir in os.listdir(model_path):
                if not dtype_dir.startswith('.'):
                    full_path = os.path.join(model_path, dtype_dir)
                    files.extend([os.path.join(full_path, file) for file in os.listdir(full_path) \
                                  if not file.startswith('.')])
                    
    return files



def extract_all_human_eval_filenames(category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to all benchmarks of HumanEval.

    Parameters
    ----------
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.
    sort_on_results : bool, optional
        If `True` and `category='completions'`, will keep only files from benchmarks for which the 'results' 
        folder does not exist. By default True.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')
    
    human_eval_benchmarks = []
    for benchmark in os.listdir(utils.RESULTS_FOLDER):
        if not benchmark.startswith('.') and 'HumanEval' in benchmark:
            human_eval_benchmarks.append(benchmark)

    # Only keep those that were not already processed
    if category == 'completions' and only_unprocessed:
        human_eval_benchmarks = [folder for folder in human_eval_benchmarks if 'results' not in \
                                 os.listdir(os.path.join(utils.RESULTS_FOLDER, folder))]
    # Only keep those that were already processed
    if category == 'results':
        human_eval_benchmarks = [folder for folder in human_eval_benchmarks if 'results' in \
                                 os.listdir(os.path.join(utils.RESULTS_FOLDER, folder))]
        
    files = []
    for benchmark in human_eval_benchmarks:
        files.extend(extract_human_eval_filenames(benchmark, category=category))

    return files

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import utils, datasets
from TextWiz.textwiz import loader

DATASETS = ('CyberSecEval_instruct', 'CyberSecEval_instruct_llama3', 'CyberSecEval_instruct_llama3_python')
CATEGORIES = ('completions', 'results')

NAME_TO_DATASET_MAPPING = {
    'CyberSecEval_instruct': datasets.CyberSecEvalInstruct,
    'CyberSecEval_instruct_llama3': datasets.CyberSecEvalInstructLlama3,
    'CyberSecEval_instruct_llama3_python': datasets.CyberSecEvalInstructLlama3_PythonOnly,
}

def get_folder(dataset: str, model_name: str, dtype_category: str) -> str:
    """Return the folder upon which to save the results of the given CyberSecEval benchmark.

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
    """Parse a filename corresponding to a CyberSecEval result file, and return the attributes that were used
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


def extract_filenames(dataset: str, category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to the given benchmark.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
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
    """Return all filenames corresponding to all benchmarks of CyberSecEval.

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

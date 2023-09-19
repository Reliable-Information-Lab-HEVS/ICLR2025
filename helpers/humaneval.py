"""This file provides helper functions to work with the HumanEval results.
"""

import os
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from engine import loader
from helpers import utils
from helpers import datasets

CATEGORIES = ['completions', 'results']


def get_folder(prompt_template_mode: str, model_name: str, dtype_category: str,
                         instruct: bool = False, use_context: bool = False) -> str:
    """Return the folder upon which to save the results of the given HumanEval benchmark.

    Parameters
    ----------
    prompt_template_mode : str
        The mode for the prompt template.
    model_name : str
        The model name.
    dtype_category : str
        Dtype used by the model.
    instruct : bool, optional
        Whether the benchmark is Instruct or not, by default False
    use_context : bool, optional
        Whether to use context or not (only if `instruct=True`), by default False

    Returns
    -------
    str
        Folder where we save the benchmark results.
    """
    
    if instruct:
        path = os.path.join(utils.RESULTS_FOLDER , f'HumanEvalInstruct_{prompt_template_mode}_{use_context}',
                            'completions', model_name, dtype_category)
    else:
        path = os.path.join(utils.RESULTS_FOLDER , f'HumanEval_{prompt_template_mode}', 'completions', model_name,
                            dtype_category)
        
    return path



def parse_filename(filename: str) -> dict:
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



def extract_filenames(benchmark: str, category: str = 'completions') -> list[str]:
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



def extract_all_filenames(category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to all benchmarks of HumanEval.

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
        files.extend(extract_filenames(benchmark, category=category))

    return files



def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))



def pass_at_k(num_samples: int | list[int] | np.ndarray, num_correct: list[int] | np.ndarray,
                       k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array.

    Parameters
    ----------
    num_samples : int | list[int] | np.ndarray
        Number of sample solutions per problem.
    num_correct : list[int] | np.ndarray
        Number of correct programs per problem.
    k : int
        Value k.

    Returns
    -------
    np.ndarray
        pass@k for each problem.
    """

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])



def evaluate_pass_at_k(result_file: str, k: list[int] = [1, 10, 100]) -> dict:
    """Evaluate the pass@k of a HumanEval benchmark for different `k`, if it is possible to compute it.

    Parameters
    ----------
    result_file : str
        File where the results of the benchmark are stored.
    k : list[int], optional
        The different `k` for pass@k, by default [1, 10, 100]

    Returns
    -------
    dict
        The different pass@k for each k.

    """

    results = defaultdict(list)

    for sample in utils.load_jsonl(result_file):
        results[sample["task_id"]].append(sample)

    if len(results) != len(datasets.HumanEval()):
        raise RuntimeError('Some problems are not attempted.')
    
    first_value_length = len(next(iter(results.values())))
    if not all(first_value_length == l for l in map(len, results.values())):
        raise RuntimeError('Not all problems have been solved the same number of times.')

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [x["passed"] for x in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    out = {f"pass@{k}": pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    return out



def all_passes_at_k(k: list[int] = [1, 10, 100]) -> list[dict]:
    """Compute the pass@k for all available HumanEval benchmarks.

    Parameters
    ----------
    k : list[int], optional
        The different `k` for pass@k, by default [1, 10, 100]

    Returns
    -------
    list[dict]
        A list of dictionary containing all details about the given benchmark, and the pass@k.
    """

    files = extract_all_filenames(category='results')

    # Compute the pass at k for every result file
    passes = []
    for file in files:

        attributes = parse_filename(file)
        # Discard unnecessary keys
        attributes.pop('category', None)
        attributes.pop('associated_results_folder', None)
        attributes.pop('associated_results_file', None)
        attributes.pop('associated_completions_file', None)

        model = attributes['model']
        model_size = loader.ALL_MODELS_PARAMS[model]
        model_family = loader.ALL_MODELS_FAMILY[model]

        pass_at_k = evaluate_pass_at_k(file, k)
        passes.append({'model_size': model_size, 'model_family': model_family, **attributes, **pass_at_k})

    return passes



def model_wise_pass_at_k(k: int = 1, nan_values: str | None = '-') -> pd.DataFrame:
    """Compute the pass@k model-wise for all benchmarks, i.e. compare the benchmarks for each model.

    Parameters
    ----------
    k : int, optional
        The `k` in pass@k, by default 1
    nan_values : str | None, optional
        Representation of missing values, by default '-'

    Returns
    -------
    pd.DataFrame
        The model-wise pass@k.
    """

    results = all_passes_at_k([k])
        
    # Swap to a "by model view"
    models = set([x['model'] for x in results])

    dics = []
    for model in models:

        # Find all results corresponding to current model
        model_results = [x for x in results if x['model'] == model]

        model_family = model_results[0]['model_family']
        model_size = model_results[0]['model_size']
        dic = {'model': model, 'model_family': model_family, 'model_size': model_size}

        for res in model_results:
            dataset = res['dataset']
            if dataset == 'HumanEval':
                benchmark = res['mode']
            else:
                benchmark = res['mode'] + '_' + str(res['use_context'])

            dic[(dataset, benchmark)] = res[f'pass@{k}']*100

        dics.append(dic)

    # sort first according to model_family, then model size
    df = pd.DataFrame.from_records(dics, index='model').sort_values(['model_family', 'model_size'])
    df = df.drop(columns=['model_family', 'model_size'])
    # sort columns according to dataset, then put "generation" before "default"
    columns = sorted(df.columns.to_list(), key=lambda x: (x[0], -len(x[1])))
    df = df[columns]

    # use multi-column index
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if nan_values is not None:
        df = df.fillna(nan_values)

    return df



def latex(df: pd.DataFrame, **kwargs):
    """Print a DataFrame as a latex table with some default arguments.
    """
    table = df.to_latex(float_format=lambda s: f'${s:.1f}$', escape=True, position='H', na_rep='-',
                        column_format='l|' + 'c'*len(df.columns), multicolumn_format='c',
                        index_names=False, **kwargs)
    print(table)
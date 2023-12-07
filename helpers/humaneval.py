"""This file provides helper functions to work with the HumanEval results.
"""

import os
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from helpers import utils
from helpers import datasets


CATEGORIES = ['completions', 'results']
DATASETS = list(datasets.HUMANEVAL_DATASETS_MAPPING.keys())


def get_folder(dataset: str, prompt_template_mode: str, model_name: str, dtype_category: str,
               use_context: bool = False) -> str:
    """Return the folder upon which to save the results of the given HumanEval benchmark.

    Parameters
    ----------
    dataset : str
        The dataset used.
    prompt_template_mode : str
        The mode for the prompt template.
    model_name : str
        The model name.
    dtype_category : str
        Dtype used by the model.
    use_context : bool, optional
        Whether to use context or not (only if `instruct=True`), by default False

    Returns
    -------
    str
        Folder where we save the benchmark results.
    """

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')
    
    if dataset == 'HumanEvalInstruct':
        path = os.path.join(utils.RESULTS_FOLDER , dataset, f'{prompt_template_mode}_{use_context}',
                            'completions', model_name, dtype_category)
    else:
        path = os.path.join(utils.RESULTS_FOLDER , dataset, f'{prompt_template_mode}', 'completions', model_name,
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

    # The file format is: utils.RESULTS_FOLDER/dataset/mode/category/model/dtype/temperature.jsonl

    _, dataset, mode, category, model, dtype, temperature_name = filename.rsplit('/', 6)

    associated_results_folder = os.path.join(utils.RESULTS_FOLDER, dataset, mode, 'results')
    associated_results_file = os.path.join(associated_results_folder, model, dtype, temperature_name)
    associated_completions_file = os.path.join(utils.RESULTS_FOLDER, dataset, mode, 'completions', model, dtype,
                                               temperature_name)
    
    temperature = float(temperature_name.split('_', 1)[1].rsplit('.', 1)[0])
    benchmark_name = dataset + '_' + mode
    if dataset == 'HumanEvalInstruct':
        mode, use_context = mode.split('_', 1)
    else:
        use_context = None

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



def extract_filenames(dataset: str, mode: str, category: str = 'completions') -> list[str]:
    """Return all filenames corresponding to a HumanEval benchmark.

    Parameters
    ----------
    dataset : str
        The name of the dataset, i.e. `HumanEval` or `HumanEvalInstruct`.
    mode : str
        The template mode used for the benchmark, e.g. `default` for HumanEval, or `default_True` for HumanEvalInstruct.
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    # The file format is: utils.RESULTS_FOLDER/dataset/mode/category/model/dtype/temperature.jsonl

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, dataset, mode, category)

    if not os.path.isdir(benchmark_path):
        raise RuntimeError('The path to the current benchmark does not exist.')
    
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
    

    files = []

    existing_datasets = [dataset for dataset in os.listdir(utils.RESULTS_FOLDER) if dataset in DATASETS]
    for dataset in existing_datasets:
        dataset_path = os.path.join(utils.RESULTS_FOLDER, dataset)
        for mode in os.listdir(dataset_path):
            if not mode.startswith('.'):
                
                try:
                    existing_files = extract_filenames(dataset, mode, category=category)
                except RuntimeError:
                    continue

                # Only keep those that were not already processed
                if category == 'completions' and only_unprocessed:
                    # Add it only if corresponding results file does not already exist
                    for file in existing_files:
                        if not os.path.exists(file.replace('completions', 'results')):
                            files.append(file)
                else:
                    files.extend(extract_filenames(dataset, mode, category=category))

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

    dataset = parse_filename(result_file)['dataset']

    for sample in utils.load_jsonl(result_file):
        results[sample["task_id"]].append(sample)

    if len(results) != len(datasets.HUMANEVAL_DATASETS_MAPPING[dataset]()):
        raise RuntimeError(f'Some problems are not attempted for file {result_file}.')
    
    first_value_length = len(next(iter(results.values())))
    if not all(first_value_length == l for l in map(len, results.values())):
        raise RuntimeError(f'Not all problems have been solved the same number of times in file {result_file}.')

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



def find_error_causes(result_file: str) -> tuple[list, list]:
    """Retrieve the different causes of error (exception types) for which the code failed.

    Parameters
    ----------
    result_file : str
        File where the results are stored.

    Returns
    -------
    tuple[list, list]
        The different exception types, and their proportion.
    """

    dics = utils.load_jsonl(result_file)
    results = [dic['exception'] if dic['exception'] is not None else 'passed' for dic in dics]
    
    errors, count = np.unique(results, return_counts=True)
    # Make it the proportion instead of raw numbers
    count = count / len(results)

    return errors.tolist(), count.tolist()



def find_best_temperature_file(folder: str, k: int = 1, greedy: bool = False) -> str:
    """Find the temperature (file) corresponding to the best pass@k inside the `folder`.

    Parameters
    ----------
    folder : str
        Folder in which we store the results corresponding to different temperatures.
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default False

    Returns
    -------
    str
        The file corresponding to the results with the nest temperature.
    """

    if greedy and k != 1:
        raise ValueError('If you want to compute pass@k with `k != 1`, you must set `greedy=False`.')
    
    greedy_file = os.path.join(folder, 'temperature_0.0.jsonl')
    if greedy:
        if not os.path.exists(greedy_file):
            raise RuntimeError('Looks like we never computed completions with temperature 0.')
        else:
            # This is only used as a check to raise if somehow the file is not complete
            _ = evaluate_pass_at_k(greedy_file, [k])
            return greedy_file
    
    else:
        temperature_files = [os.path.join(folder, file) for file in os.listdir(folder) \
                              if file.startswith('temperature') and file != greedy_file]

        passes = []
        for file in temperature_files:
            try:
                passes.append(evaluate_pass_at_k(file, [k])[f'pass@{k}'])
            except RuntimeError:
                continue
            except KeyError:
                continue
        
        if len(passes) == 0:
            raise RuntimeError(f'The k you provided is larger than the number of sequences generated for any file in {folder}')
        
        return temperature_files[np.argmax(passes)]
    


def _get_default_dtype(model_name: str) -> str:
    """Return the default dtype used by a given model.
    """
    from TextWiz.textwiz import loader

    if model_name == 'bloom-176B':
        return 'int8'
    else:
        return str(loader.get_model_dtype(model_name)).split('.', 1)[1]



def find_folders_with_dtype(dataset: str, mode: str, dtype: str = 'default') -> list[str]:
    """Return the folders corresponding to the given `dataset`, `mode`, and `dtype`, in which we can find the 
    results with different temperatures.

    Parameters
    ----------
    dataset : str
        The name of the dataset, i.e. `HumanEval` or `HumanEvalInstruct`.
    mode : str
        The template mode used for the benchmark, e.g. `default` for HumanEval, or `default_True` for HumanEvalInstruct.
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'

    Returns
    -------
    list[str]
        All the folders corresponding to the given `dataset`, `mode`, and `dtype`.
    """

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')

    dtypes = ['default', 'int8', 'int4']
    if dtype not in dtypes:
        raise ValueError(f'Dtype must be in {*dtypes,}')

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, dataset, mode, 'results')
    
    folders = []
    for model in os.listdir(benchmark_path):
        if not model.startswith('.'):
            model_path = os.path.join(benchmark_path, model)
            dtype_dirs = [x for x in os.listdir(model_path) if not x.startswith('.')]
            # Get the default dtype of the current model
            actual_dtype = _get_default_dtype(model) if dtype == 'default' else dtype
            # If dtype is present in the list of dtype, add the folder to the list
            if actual_dtype in dtype_dirs:
                folders.append(os.path.join(model_path, actual_dtype))

    return folders



def benchmark_passes_at_k_and_error_causes(dataset: str, mode: str, dtype: str = 'default', k: int = 1,
                                           greedy: bool = True, to_df: bool = False) -> list[dict] | pd.DataFrame:
    """Compute the pass@k (for the best temperature) and error causes for all models corresponding to the given
    `dataset`, `mode`, and `dtype`. 

    Parameters
    ----------
    dataset : str
        The name of the dataset, i.e. `HumanEval` or `HumanEvalInstruct`.
    mode : str
        The template mode used for the benchmark, e.g. `default` for HumanEval, or `default_True` for HumanEvalInstruct.
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True
    to_df : bool, optional
        Whether to convert to DataFrame or not, by default False

    Returns
    -------
    list[dict] | pd.DataFrame
        A list of dictionary or DataFrame containing all interesting attributes for each model in the benchmark.
    """

    from TextWiz.textwiz import loader
    
    folders = find_folders_with_dtype(dataset, mode, dtype=dtype)

    results = []
    for folder in folders:
        # Try to find the best temperature file for given k
        try:
            file = find_best_temperature_file(folder, k=k, greedy=greedy)
        # It is impossible to find for current model and dtype -> continue to next model
        except RuntimeError:
            continue

        attributes = parse_filename(file)
        # Discard unnecessary keys
        attributes.pop('category', None)
        attributes.pop('associated_results_folder', None)
        attributes.pop('associated_results_file', None)
        attributes.pop('associated_completions_file', None)

        model = attributes['model']
        model_size = loader.ALL_MODELS_PARAMS[model]
        model_family = loader.ALL_MODELS_FAMILY[model]

        pass_at_k_ = evaluate_pass_at_k(file, [k])
        error_causes, error_proportions = find_error_causes(file)
        results.append({'model_size': model_size, 'model_family': model_family, 'error_causes': error_causes,
                       'error_proportions': error_proportions, **attributes, **pass_at_k_})
        
    # sort them according to family, then size, then name (for model with same family and size)
    results = sorted(results, key=lambda x: (x['model_family'], x['model_size'], x['model']))

    if to_df:
        return pd.DataFrame.from_records(results)
    else:
        return results



def all_passes_at_k_and_error_causes(dtype: str = 'default', k: int = 1, greedy: bool = True,
                                     to_df: bool = False) -> dict[str, list[dict] | pd.DataFrame]:
    """Compute the pass@k and error causes (for the best temperature) for all models and all benchmarks
    corresponding to the given `dtype`. 

    Parameters
    ----------
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True
    to_df : bool, optional
        Whether to convert to DataFrame or not, by default False

    Returns
    -------
    dict[str, list[dict] | pd.DataFrame]
        A dictionary mapping each benchmark to its results.
    """

    benchmarks = {}
    for dataset in DATASETS:
        dataset_path = os.path.join(utils.RESULTS_FOLDER, dataset)
        for mode in os.listdir(dataset_path):
            if not mode.startswith('.'):
                benchmarks[dataset + '_' + mode] = benchmark_passes_at_k_and_error_causes(dataset, mode, dtype=dtype,
                                                                                          k=k, greedy=greedy, to_df=to_df)

    return benchmarks



def model_wise_pass_at_k(dtype: str = 'default', k: int = 1, greedy: bool = True,
                         nan_values: str | None = '-') -> pd.DataFrame:
    """Comparison of the pass@k for all the models and benchmarks, and given `dtype`.

    Parameters
    ----------
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True
    nan_values : str | None, optional
        Optional str to fill missing values, by default '-'

    Returns
    -------
    pd.DataFrame
        A multi-column dataframe with the results.
    """

    from TextWiz.textwiz import loader
    
    benchs = all_passes_at_k_and_error_causes(dtype=dtype, k=k, greedy=greedy, to_df=True)

    # Keep only needed columns and set the name as a tuple for later use MultiIndex
    new_dfs = []
    for df in benchs.values():

        if df.empty:
            continue

        dataset = df['dataset'][0]
        mode = df['mode'][0]
        context = str(df['use_context'][0])

        if dataset == 'HumanEvalInstruct':
            subindex = mode + '_' + context
        else:
            subindex = mode

        new_df = df[['model', f'pass@{k}']].copy()
        new_df[f'pass@{k}'] = new_df[f'pass@{k}']*100
        new_df.rename(columns={f'pass@{k}': (dataset, subindex)}, inplace=True)
        new_dfs.append(new_df)

    # Merge dataframes depending on size
    if len(new_dfs) == 0:
        return pd.DataFrame()
    elif len(new_dfs) == 1:
        final_df = new_dfs[0]
    else:
        final_df = new_dfs[0]
        for df in new_dfs[1:]:
            final_df = final_df.merge(df, on='model', how='outer')

    # Add the family and size to be able to sort after all merging (cannot keep the entries during merging)
    final_df['model_family'] = [loader.ALL_MODELS_FAMILY[model] for model in final_df['model']]
    final_df['model_size'] = [loader.ALL_MODELS_PARAMS[model] for model in final_df['model']]
    # Sort and remove the family and size
    final_df = final_df.sort_values(['model_family', 'model_size', 'model']).drop(columns=['model_family', 'model_size'])
    # Set the model as index
    final_df = final_df.set_index('model')

    # sort columns according to dataset, then put "generation" before "default"
    columns = sorted(final_df.columns.to_list(), key=lambda x: (x[0], -len(x[1])))
    final_df = final_df[columns]
    # use multi-column index
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)

    if nan_values is not None:
        final_df = final_df.fillna(nan_values)

    return final_df



def model_wise_best_score(dtype: str = 'default', k: int = 1, greedy: bool = True) -> pd.DataFrame:
    """Return the best pass@k for each model across all available benchmarks for the given `dtype` and `k`.

    Parameters
    ----------
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame containing the best pass@k.
    """

    df = model_wise_pass_at_k(dtype=dtype, k=k, greedy=greedy, nan_values=None)

    if df.empty:
        return pd.DataFrame()
    
    # Initialize result
    out = pd.DataFrame(index=df.index.copy())
    out[f'best_pass@{k}'] = df.max(axis=1, skipna=True)
    out['best_benchmark'] = df.idxmax(axis=1, skipna=True).map(lambda idx: '_'.join(idx))

    return out



def model_wise_error_causes(dtype: str = 'default', k: int = 1, greedy: bool = True, save: bool = False,
                            model_subset: list | None = None, pretty_names_mapping: dict | None = None,
                            title: bool = True):
    """Plot the model-wise error causes for each benchmark, for the given `dtype` and `k`. Note that `k` is only
    used to select which temperature gave the best results if `greedy=False`.

    Parameters
    ----------
    dtype : str, optional
        A precise dtype to use for fetching the results, by default 'default'
    k : int, optional
        The `k` in pass@k, by default 1. Only used to select which temperature gave the best results for a given
        model if `greedy=False`.
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True
    save : bool, optional
        Whether to save the plots or not, by default False
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    from helpers import plot_config

    benchs = all_passes_at_k_and_error_causes(dtype=dtype, k=k, greedy=greedy, to_df=False)

    figs = []

    for benchmark in benchs:

        records = benchs[benchmark]

        if model_subset is not None:
            records = [record for record in records if record['model'] in model_subset]

        if len(records) == 0:
            continue

        all_errors = []
        for record in records:
            all_errors.extend(record['error_causes'])

        possible_errors = set(all_errors)
        # Reorder
        possible_errors.discard('passed')
        possible_errors.discard('AssertionError')
        possible_errors = ['passed', 'AssertionError'] + sorted(list(possible_errors))
        # Convert to numpy
        possible_errors = np.array(possible_errors)

        error_matrix = np.zeros((len(records), len(possible_errors)))
        models = []

        for i, record in enumerate(records):
            models.append(record['model'])
            errors = record['error_causes']
            proportions = record['error_proportions']
            for error, proportion in zip(errors, proportions):
                index = np.nonzero(error == possible_errors)[0][0]
                error_matrix[i, index] = proportion


        # Mask to avoid plotting cells with 0s
        mask = np.where(error_matrix == 0, 1, 0)

        if pretty_names_mapping is not None:
            pretty_names = [pretty_names_mapping[model] for model in models]
            ylabels = pretty_names
        else:
            ylabels = models

        size = np.array(plt.rcParams['figure.figsize']) * 2

        plt.figure(figsize=size)
        if title:
            plt.title(f'{benchmark}, dtype={dtype}, k={k}')
        sns.heatmap(error_matrix, mask=mask, annot=True, annot_kws={'fontsize': 'x-small'}, fmt='.2f',
                    xticklabels=possible_errors, yticklabels=ylabels, cbar=True, cmap='Blues')
        plt.xticks(rotation=45, ha='right')
        # Set background color for masked values
        ax = plt.gca()
        ax.set_facecolor('lightyellow')
        # ax.set_facecolor('lightgoldenrodyellow')

        if save:
            plt.savefig(os.path.join(utils.ROOT_FOLDER, 'plots', f'{benchmark}_{k}_{dtype}.pdf'), bbox_inches='tight')

        figs.append(plt.gcf())

    return figs



def model_wise_pass_at_k_comparison(dtypes: list[str] = ['default', 'int8'], k: int = 1, greedy: bool = True,
                                    nan_values: str | None = '-') -> pd.DataFrame:
    """Compare the pass@k performances between different dtypes.

    Parameters
    ----------
    dtypes : list[str], optional
        The dtypes to compare, by default ['default', 'int8']
    k : int, optional
        The `k` in pass@k, by default 1
    greedy : bool, optional
        Whether we are computing pass@k for greedy decoding or not, by default True
    nan_values : str | None, optional
        Optional str to fill missing values, by default '-'

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results.
    """
    
    dfs = []
    for dtype in dtypes:
        dfs.append(model_wise_pass_at_k(dtype=dtype, k=k, greedy=greedy, nan_values=nan_values))

    common_cols = set(dfs[0].columns.copy())
    for df in dfs[1:]:
        common_cols = common_cols.intersection(set(df.columns.copy()))

    common_cols = list(common_cols)
    final_df = dfs[0][common_cols]
    i = 0
    for df in dfs[1:]:
        final_df = final_df.merge(df, how='inner', on='model', suffixes=[dtypes[i], dtypes[i+1]])
        i += 1

    columns = []
    for column in final_df.columns.copy():
        for dtype in dtypes:
            if dtype in column[0]:
                new_column = (column[0].replace(dtype, ''), column[1], dtype)
                columns.append(new_column)
                break

    final_df.columns = columns
    sorted_columns = sorted(columns, key=lambda x: (x[0], -len(x[1]), x[2]))
    final_df = final_df[sorted_columns]
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)

    return final_df


PRETTY_NAMES_MAPPING = {
    'code-llama-34B': 'CodeLlama 34B',
    'code-llama-34B-instruct': 'CodeLlama - Instruct 34B',
    'code-llama-34B-python': 'CodeLlama - Python 34B',
    'codegen-16B': 'CodeGen - Mono 16B',
    'codegen25-7B': 'CodeGen2.5 - Mono 7B',
    'codegen25-7B-instruct': 'CodeGen2.5 - Instruct 7B',
    'llama2-70B': 'Llama2 70B',
    'llama2-70B-chat': 'Llama2 - Chat 70B',
    'star-chat-alpha': 'StarChat (alpha)',
    'star-coder': 'StarCoder',
    'star-coder-base': 'StarCoderBase',
}

MODELS = [k for k in PRETTY_NAMES_MAPPING.keys()]
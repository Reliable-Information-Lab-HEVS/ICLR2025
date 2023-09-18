import os
import copy
import shutil

from helpers import utils, datasets
from engine import stopping, loader
import human_eval


# path = os.path.join(utils.RESULTS_FOLDER, 'HumanEval_completions')
# models = [os.path.join(path, dir) for dir in os.listdir(path) if not dir.startswith('.')]
# files_in = []
# files_out = []
# for model_dir in models:
#     files_in.extend([os.path.join(model_dir, file) for file in os.listdir(model_dir)])
#     files_out.extend([os.path.join(model_dir, file.rsplit('.', 1)[0] + '_modified.jsonl') for file in os.listdir(model_dir)])

# problems = datasets.HumanEval().samples_by_id()

# for file_in, file_out in zip(files_in, files_out):
#     completions = utils.load_jsonl(file_in)
#     new_completions = []
#     for completion in completions:
#         new_completion = copy.deepcopy(completion)
#         new_completion['completion'] = stopping.post_process_sequences([completion['completion']])[0]
#         new_completions.append(new_completion)

#     utils.save_jsonl(new_completions, file_out)






# human_eval_folders = [os.path.join(utils.RESULTS_FOLDER, folder) for folder in os.listdir(utils.RESULTS_FOLDER) if
#                       (not folder.startswith('.') and 'HumanEval' in folder)]
# # Only keep those that were not already processed
# human_eval_folders = [folder for folder in human_eval_folders if 'completions' in os.listdir(folder)]

# # This contains utils.RESULTS_FOLDER/HumanEval.../completions
# human_eval_completions = [os.path.join(folder, 'completions') for folder in human_eval_folders]

# files = []
# for benchmark in human_eval_completions:
#     for model_dir in os.listdir(benchmark):
#         if not model_dir.startswith('.'):
#             full_path = os.path.join(benchmark, model_dir)
#             files.extend([os.path.join(full_path, file) for file in os.listdir(full_path) if not file.startswith('.')])

# chat_files = []
# for file in files:
#     dics = utils.load_jsonl(file)
#     # print(dics)
#     if 'model_output' in dics[0].keys():
#         chat_files.append(file)

# print(chat_files)


# samples = datasets.HumanEval().samples_by_id()
# for file in chat_files:
#     dics = utils.load_jsonl(file)
#     new_dics = []
#     for dic in dics:
#         sample = samples[dic['task_id']]
#         new_completion = human_eval.extract_completions([dic['model_output']], sample)[0]
#         new_dic = {'task_id':dic['task_id'], 'model_output':dic['model_output'], 'completion': new_completion}
#         new_dics.append(new_dic)
    
#     utils.save_jsonl(new_dics, file)


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

    # The file format is: utils.RESULTS_FOLDER/benchmark/category/model/temperature.jsonl

    assert 'HumanEval' in benchmark, 'The benchmark name is not correct'

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, benchmark, category)
    
    files = []
    for model_dir in os.listdir(benchmark_path):
        if not model_dir.startswith('.'):
            model_path = os.path.join(benchmark_path, model_dir)
            files.extend([os.path.join(model_path, file) for file in os.listdir(model_path) \
                        if not file.startswith('.')])
                    
    return files



def extract_all_human_eval_filenames(category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
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


def dtype_category(model, quantization_4bits = False, quantization_8bits = False) -> str:
    """Return a string representation of the model dtype."""
    if quantization_4bits:
        return 'int4'
    elif quantization_8bits:
        return 'int8'
    else:
        return str(loader.get_model_dtype(model)).split('.', 1)[1]


if __name__ == '__main__':

    # The file format is: utils.RESULTS_FOLDER/benchmark/category/model/temperature.jsonl

    files = extract_all_human_eval_filenames(category='completions', only_unprocessed=False)

    for file in files:

        if os.path.isdir(file):
            continue

        else:

            folder, model, name = file.rsplit('/', 2)
            if model == 'bloom-176B':
                dtype = dtype_category(model, quantization_8bits=True)
            else:
                dtype = dtype_category(model)
            
            new_file = os.path.join(folder, model, dtype, name)
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            shutil.move(file, new_file)
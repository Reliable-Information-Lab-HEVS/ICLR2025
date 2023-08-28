import os
from tqdm import tqdm

from code_execution import evaluation
from helpers import utils

# Files are at: utils.RESULTS_FOLDER/HumanEval.../completions/model/temperature.jsonl

human_eval_folders = [os.path.join(utils.RESULTS_FOLDER, folder) for folder in os.listdir(utils.RESULTS_FOLDER) if
                      (not folder.startswith('.') and 'HumanEval' in folder)]
# Only keep those that were not already processed
human_eval_folders = [folder for folder in human_eval_folders if ('results' not in os.listdir(folder) and 'completions' in os.listdir(folder))]

# This contains utils.RESULTS_FOLDER/HumanEval.../completions
human_eval_completions = [os.path.join(folder, 'completions') for folder in human_eval_folders]

files = []
for benchmark in human_eval_completions:
    for model_dir in os.listdir(benchmark):
        if not model_dir.startswith('.'):
            full_path = os.path.join(benchmark, model_dir)
            files.extend([os.path.join(full_path, file) for file in os.listdir(full_path) if not file.startswith('.')])


if __name__ == '__main__':
    result_files = []
    for file in tqdm(files):
        out_file = evaluation.evaluate_functional_correctness(file, n_workers=6, timeout=3)
        result_files.append(out_file)

    # Format of the results is utils.RESULTS_FOLDER/HumanEval/results/model/temperature.jsonl
    result_folders = set([file.rsplit('/', 2)[0] for file in result_files])
    # Extract the relative paths (relative to current dir, i.e. utils.ROOT_FOLDER)
    result_folders = [os.path.relpath(path) for path in result_folders]
    # This is used to determine which files we need to copy back to the host outside of the docker instance
    utils.save_txt(result_folders, 'folders_to_copy.txt')

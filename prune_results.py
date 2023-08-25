import os
import copy

from helpers import utils, datasets
from engine import stopping
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


human_eval_folders = [os.path.join(utils.RESULTS_FOLDER, folder) for folder in os.listdir(utils.RESULTS_FOLDER) if
                      (not folder.startswith('.') and 'HumanEval' in folder)]
# Only keep those that were not already processed
human_eval_folders = [folder for folder in human_eval_folders if 'completions' in os.listdir(folder)]

# This contains utils.RESULTS_FOLDER/HumanEval.../completions
human_eval_completions = [os.path.join(folder, 'completions') for folder in human_eval_folders]

files = []
for benchmark in human_eval_completions:
    for model_dir in os.listdir(benchmark):
        if not model_dir.startswith('.'):
            full_path = os.path.join(benchmark, model_dir)
            files.extend([os.path.join(full_path, file) for file in os.listdir(full_path) if not file.startswith('.')])

chat_files = []
for file in files:
    dics = utils.load_jsonl(file)
    # print(dics)
    if 'model_output' in dics[0].keys():
        chat_files.append(file)

print(chat_files)


samples = datasets.HumanEval().samples_by_id()
for file in chat_files:
    dics = utils.load_jsonl(file)
    new_dics = []
    for dic in dics:
        sample = samples[dic['task_id']]
        new_completion = human_eval.extract_completions([dic['model_output']], sample)[0]
        new_dic = {'task_id':dic['task_id'], 'model_output':dic['model_output'], 'completion': new_completion}
        new_dics.append(new_dic)
    
    utils.save_jsonl(new_dics, file)


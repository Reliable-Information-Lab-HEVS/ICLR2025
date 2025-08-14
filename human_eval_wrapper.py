import argparse
import time

import torch

from TextWiz import textwiz
from TextWiz.textwiz.templates.prompt_template import PROMPT_MODES
from helpers import datasets
from helpers import utils

import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HumanEval benchmark')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--big_models', action='store_true',
                        help='If given, run the benchmark on large models that do not fit on a single gpu.')
    parser.add_argument('--big_models_only', action='store_true',
                        help='If given, only run the benchmark on large models that do not fit on a single gpu.')
    parser.add_argument('--special_only', action='store_true',
                        help='If given, will only run the benchmark on models with a non-default prompt template.')
    parser.add_argument('--mode', type=str, default='generation', choices=PROMPT_MODES,
                        help='The mode for the prompt template.')
    parser.add_argument('--new_models', action='store_true',
                        help='If given, will only run the benchmark on new models (not in the original HumanEval paper).')
    parser.add_argument('--instruct', action='store_true',
                        help='If given, run the HumanEvalInstruct benchmark.')
    parser.add_argument('--no_context', action='store_false',
                        help='If given, do NOT use the context in the HumanEvalInstruct benchmark.')
    parser.add_argument('--language', type=str, default='py',
                        help='If given, run the corresponding benchmark.')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    big_models = args.big_models
    big_models_only = args.big_models_only
    special_only = args.special_only
    new_models = args.new_models
    instruct = args.instruct
    mode = args.mode
    use_context = args.no_context
    language = args.language

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')
    
    if instruct and language != 'py':
        raise ValueError('instruct and language options are mutually exclusive.')
    
    if language not in datasets.MULTIPLE_LANGUAGE_MAPPING.keys() and language != 'py':
        raise ValueError(f'The language must be one of {*datasets.MULTIPLE_LANGUAGE_MAPPING.keys(),}.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models (only keep the good coders)
    new_models = textwiz.NEW_CODERS
    small_models = textwiz.SMALL_GOOD_CODERS_SPECIAL_PROMPT if special_only else textwiz.SMALL_GOOD_CODERS
    large_models = textwiz.LARGE_GOOD_CODERS_SPECIAL_PROMPT if special_only else textwiz.LARGE_GOOD_CODERS
    if big_models_only:
        models = large_models
    elif big_models:
        models = small_models + large_models
    elif new_models:
        models = new_models
    else:
        models = small_models

    print(f'Launching computations with {num_gpus} gpus available.')

    # Create the commands to run
    gpu_footprints = textwiz.estimate_number_of_gpus(models, int8, int4)
    commands = [f'python3 -u human_eval.py {model} --language {language} --mode {mode}' for model in models]
    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]
    if instruct:
        commands = [c + ' --instruct' for c in commands]
    if not use_context:
        commands = [c + ' --no_context' for c in commands]
        
    t0 = time.time()

    commands = [
        'python3 -u human_eval.py star-chat-2-instruct --language py --mode generation',
        'python3 -u human_eval.py code-llama-34B --language py --mode generation',
        'python3 -u human_eval.py code-llama-34B-python --language py --mode generation',
        'python3 -u human_eval.py code-llama-34B-instruct --language py --mode generation',
        'python3 -u human_eval.py code-llama-34B-instruct --language py --mode default',
        'python3 -u human_eval.py code-llama-34B --language py --mode generation --instruct',
        'python3 -u human_eval.py code-llama-34B-python --language py --mode generation --instruct',
        'python3 -u human_eval.py code-llama-34B-instruct --language py --mode generation --instruct',
        'python3 -u human_eval.py code-llama-34B-instruct --language py --mode default --instruct',
        'python3 -u human_eval.py code-llama-34B --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B-python --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B-instruct --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B-python --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B-instruct --language php --mode generation',
        'python3 -u human_eval.py code-llama-34B --language rs --mode generation',
        'python3 -u human_eval.py code-llama-34B-python --language rs --mode generation',
        'python3 -u human_eval.py code-llama-34B-instruct --language rs --mode generation',

    ]

    #utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)

    #run them sequentially for now
    for command in commands:
        # run them
        print(f'Running command: {command}')
        #there is no help function, so we can use os.system
        os.system(command)

    dt = time.time() - t0
    print(f'Overall it took {dt/3600:.2f}h !')


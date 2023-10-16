import os
import gc
import argparse
import time
import copy
import re

import torch

import engine
from engine import stopping
from engine.prompt_template import PROMPT_MODES
from engine.code_parser import CodeParser, PythonParser
from helpers import datasets
from helpers import utils
from helpers import humaneval




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
    parser.add_argument('--mode', type=str, default='default', choices=PROMPT_MODES,
                        help='The mode for the prompt template.')
    parser.add_argument('--instruct', action='store_true',
                        help='If given, run the HumanEvalInstruct benchmark.')
    parser.add_argument('--no_context', action='store_false',
                        help='If given, do NOT use the context in the HumanEvalInstruct benchmark.')
    parser.add_argument('--php', action='store_true',
                        help='If given, run the HumanEvalPHP benchmark.')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    big_models = args.big_models
    big_models_only = args.big_models_only
    special_only = args.special_only
    instruct = args.instruct
    mode = args.mode
    use_context = args.no_context
    php = args.php

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')
    
    if instruct and php:
        raise ValueError('instruct and php options are mutually exclusive.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models (only keep the good coders)
    small_models = engine.SMALL_GOOD_CODERS_SPECIAL_PROMPT if special_only else engine.SMALL_GOOD_CODERS
    large_models = engine.LARGE_GOOD_CODERS_SPECIAL_PROMPT if special_only else engine.LARGE_GOOD_CODERS
    if big_models_only:
        models = large_models
    elif big_models:
        models = small_models + large_models
    else:
        models = small_models

    print(f'Launching computations with {num_gpus} gpus available.')

    gpu_footprints = engine.estimate_number_of_gpus(models, int8, int4)
    commands = [f'python3 human_eval_exec {model} --mode {mode}' for model in models]
    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]
    if instruct:
        commands = [c + ' --instruct' for c in commands]
    if not use_context:
        commands = [c + ' --no_context' for c in commands]
    if php:
        commands = [c + ' --php' for c in commands]
        
    utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)


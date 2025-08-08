import argparse
import time

import torch

from TextWiz import textwiz
from helpers import utils




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AATK benchmark')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--big_models', action='store_true',
                        help='If given, run the benchmark on large models that do not fit on a single gpu.')
    parser.add_argument('--big_models_only', action='store_true',
                        help='If given, only run the benchmark on large models that do not fit on a single gpu.')
    parser.add_argument('--new_models', action='store_true',
                        help='If given, run the benchmark on new models')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    big_models = args.big_models
    big_models_only = args.big_models_only
    run_new_models = args.new_models

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models (only keep the good coders)
    small_models = textwiz.SMALL_GOOD_CODERS
    large_models = textwiz.LARGE_GOOD_CODERS
    new_models = textwiz.NEW_CODERS
    if run_new_models:
        models = new_models
    elif big_models_only:
        models = large_models
    elif big_models:
        models = small_models + large_models
    else:
        models = small_models

    print(f'Launching computations with {num_gpus} gpus available.')

    # Create the commands to run
    gpu_footprints = textwiz.estimate_number_of_gpus(models, int8, int4)
    commands = [f'python3 -u AATK.py {model}' for model in models]
    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]
        
    t0 = time.time()

    utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)

    dt = time.time() - t0
    print(f'Overall it took {dt/3600:.2f}h !')


import argparse
import time

import torch

from TextWiz import textwiz
from helpers import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AATKInstruct benchmark')
    parser.add_argument('--reformulation_model', type=str, choices=['chatGPT', 'zephyr', 'llama3'], default='chatGPT',
                        help='Version of the AATK instruct benchmark (i.e. model used for reformulations)')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--new_models', action='store_true',
                        help='If given, run the benchmark on new models')
    
    args = parser.parse_args()
    reformulation_model = args.reformulation_model
    int8 = args.int8
    int4 = args.int4
    run_new_models = args.new_models

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models
    # models = [
    #     'zephyr-7B-beta',
    #     'mistral-7B-instruct-v2',
    #     'starling-7B-beta',
    #     'star-chat-alpha',
    #     'llama3-8B-instruct',
    #     'command-r',
    #     'code-llama-34B-instruct',
    #     'llama2-70B-chat',
    #     'code-llama-70B-instruct',
    #     'llama3-70B-instruct',
    # ]

    new_models = textwiz.NEW_CODERS_CHAT
    if run_new_models:
        models = new_models


    print(f'Launching computations with {num_gpus} gpus available.')

    # Create the commands to run
    gpu_footprints = textwiz.estimate_number_of_gpus(models, int8, int4)
    commands = [f'python3 -u AATK_instruct.py {model} --reformulation_model {reformulation_model}' for model in models]
    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]
        
    t0 = time.time()

    utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)

    dt = time.time() - t0
    print(f'Overall it took {dt/3600:.2f}h !')


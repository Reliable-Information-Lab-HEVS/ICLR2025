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

TEMPERATURES = (0., 0.2,)

# We need to set top_k to 0 to deactivate top-k sampling
AATK_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 25,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

AATK_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}




def extract_completions(outputs: list[str], sample: dict, parser: CodeParser = PythonParser(),
                        stopping_patterns: tuple[str] = stopping.EXTENDED_CODE_STOP_PATTERNS) -> list[str]:
    
    code_outputs = [parser(sequence) for sequence in outputs]

    completions = []
    for output in code_outputs:

        # Check if the model repeated the function definition
        regex = r'def ' + re.escape(sample['entry_point']) + r'[^\n]*\n(.*)$'
        code_after_func_def = re.search(regex, output, re.DOTALL)

        if code_after_func_def:
            output = code_after_func_def.group(1)
            output = stopping.post_process_stopping_patterns([output], stopping_patterns)[0]
            # Remove the function definition
            if re.search(r'"""(.*?)"""', output, re.DOTALL):
                _, _, completion = output.split('"""', 2)
            else:
                completion = output
        else:
            completion = stopping.post_process_stopping_patterns([output], stopping_patterns)[0]

        completions.append(completion)

    return completions



@utils.duplicate_function_for_gpu_dispatch
def aatk_benchmark(model_name: str, prompt_template_mode: str, quantization_8bits: bool = False,
                   quantization_4bits: bool = False, temperatures: tuple[int] = TEMPERATURES,
                   generation_kwargs: dict = AATK_GENERATION_KWARGS,
                   greedy_generation_kwargs: dict = AATK_GREEDY_GENERATION_KWARGS):
    """Generate the aatk completions for different temperatures with the model `model_name` and
    save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the HumanEval benchmark, by default AATK_GENERATION_KWARGS
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the HumanEval benchmark, by default AATK_GREEDY_GENERATION_KWARGS
    """

    print(f'Starting with model {model_name}')

    # Override quantization for bloom because it's too big
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        model = engine.HFModel(model_name, quantization_8bits=True, max_fraction_gpu_0=0.9, max_fraction_gpus=0.9)
    else:
        model = engine.HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    # TODO: define this!
    stopping_patterns = None 
    folder = 'foo'

    dataset = datasets.AATK_benchmark()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            id = sample['id']
            prompt = sample['code']

            # In this case we use greedy decoding (the temperature parameters does not matter anymore
            # so we set it to the default which is 1)
            if temperature == 0:
                completions = [model(prompt, temperature=1., batch_size=1, stopping_patterns=stopping_patterns,
                                     prompt_template_mode=prompt_template_mode, **greedy_generation_kwargs)]
            # In this case we use top-p sampling
            else:
                completions = model(prompt, temperature=temperature, stopping_patterns=stopping_patterns,
                                    prompt_template_mode=prompt_template_mode, **generation_kwargs)

            # Save the model completions
            if model.is_chat_model():
                true_completions = extract_completions(completions, sample)
                results = [{'id': id, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]
            else:
                results = [{'id': id, 'completion': completion} for completion in completions]

            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


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
                        help='The mode for the prompt template. By default `generation`.')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    big_models = args.big_models
    big_models_only = args.big_models_only
    special_only = args.special_only
    mode = args.mode

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

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

    # arguments depending on target function
    args = (models, mode, int8, int4)

    print(f'Launching computations with {num_gpus} gpus available.')

    if num_gpus > 1:
        gpu_footprints = engine.estimate_number_of_gpus(models, int8, int4)
        utils.dispatch_jobs(gpu_footprints, num_gpus, aatk_benchmark, *args)
    else:
        for model in models:
            aatk_benchmark(*args)


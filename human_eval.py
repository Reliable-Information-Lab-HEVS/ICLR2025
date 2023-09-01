import os
import gc
import argparse
import time
import copy
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch

import engine
from engine import stopping
from engine import loader
from engine.prompt_template import PROMPT_MODES
from engine.code_parser import CodeParser, PythonParser
from helpers import datasets
from helpers import utils

# TEMPERATURES = (0., 0.2, 0.4, 0.6, 0.8, 1.)
TEMPERATURES = (0.,)

# We need to set top_k to 0 to deactivate top-k sampling
HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

HUMAN_EVAL_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

SMALL_MODELS = (
    'bloom-560M',
    'bloom-1.7B',
    'bloom-3B',
    'bloom-7.1B',
    'stable-lm-3B',
    'stable-lm-7B',
    'star-coder-base',
    'star-coder',
    'star-coder-plus',
    'star-chat-alpha',
    'star-chat-beta',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'gpt-j-6B',
    'gpt-neo-125M',
    'gpt-neo-1.3B',
    'gpt-neo-2.7B',
    'opt-125M',
    'opt-350M',
    'opt-1.3B',
    'opt-2.7B',
    'opt-6.7B',
    'opt-13B',
    'codegen-350M',
    'codegen-2B',
    'codegen-6B',
    'codegen-16B',
    'codegen2-1B',
    'codegen2-3.7B',
    'codegen2-7B',
    'codegen2-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
    'vicuna-7B',
    'vicuna-13B',
    'llama2-7B',
    'llama2-7B-chat',
    'llama2-13B',
    'llama2-13B-chat',
)

SMALL_MODELS_SPECIAL_PROMPT = (
    'star-coder-base',
    'star-coder',
    'star-coder-plus',
    'star-chat-alpha',
    'star-chat-beta',
    'codegen2-1B',
    'codegen2-3.7B',
    'codegen2-7B',
    'codegen2-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
    'vicuna-7B',
    'vicuna-13B',
    'llama2-7B-chat',
    'llama2-13B-chat',
)

LARGE_MODELS = (
    'gpt-neoX-20B',
    'opt-30B',
    'opt-66B',
    'llama2-70B',
    'llama2-70B-chat',
    'bloom-176B',
)

LARGE_MODELS_SPECIAL_PROMPT = (
    'llama2-70B-chat',
)



def extract_completions(outputs: list[str], sample: dict, parser: CodeParser = PythonParser(),
                        stopping_patterns: tuple[str] = stopping.EXTENDED_CODE_STOP_PATTERNS) -> list[str]:
    
    # code_outputs = stopping.parse_code_and_truncate(outputs, parser, stopping_patterns)
    code_outputs = [parser(sequence) for sequence in outputs]

    completions = []
    for output in code_outputs:

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
def human_eval(model_name: str, prompt_template_mode: str, temperatures: tuple[int] = TEMPERATURES,
               generation_kwargs: dict = HUMAN_EVAL_GENERATION_KWARGS,
               greedy_generation_kwargs: dict = HUMAN_EVAL_GREEDY_GENERATION_KWARGS):
    """Generate the HumanEval completions for different temperatures with the model `model_name` and
    save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the HumanEval benchmark, by default HUMAN_EVAL_GENERATION_KWARGS
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the HumanEval benchmark, by default HUMAN_EVAL_GREEDY_GENERATION_KWARGS
    """

    # Load in 8 bits for bloom due to model size
    quantization = True if model_name == 'bloom-176B' else False

    model = engine.HFModel(model_name, quantization=quantization)
    stopping_patterns = None if (model.is_chat_model and prompt_template_mode in ['default', 'chat']) else stopping.CODE_STOP_PATTERNS
    folder = os.path.join(utils.RESULTS_FOLDER , f'HumanEval_{prompt_template_mode}', 'completions', model_name)

    dataset = datasets.HumanEval()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            task_id = sample['task_id']
            prompt = sample['prompt']

            # GPT2 has only a context size of 1024, which can sometimes overflow with large `max_new_tokens`.
            if 'gpt2' in model_name:
                prompt_length = model.tokenizer.encode(prompt, return_tensors='pt').shape[-1]
                # Note that we need deepcopies to avoid changing the default values of the function inplace
                if prompt_length + generation_kwargs['max_new_tokens'] > 1024:
                    generation_kwargs = copy.deepcopy(generation_kwargs)
                    generation_kwargs['max_new_tokens'] = 1024 - prompt_length
                if prompt_length + greedy_generation_kwargs['max_new_tokens'] > 1024:
                    greedy_generation_kwargs = copy.deepcopy(greedy_generation_kwargs)
                    greedy_generation_kwargs['max_new_tokens'] = 1024 - prompt_length

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
            if model.is_chat_model and prompt_template_mode in ['default', 'chat']:
                true_completions = extract_completions(completions, sample)
                results = [{'task_id': task_id, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]
            else:
                results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()



@utils.duplicate_function_for_gpu_dispatch
def human_eval_instruct(model_name: str, prompt_template_mode: str, use_context: bool,
                        temperatures: tuple[int] = TEMPERATURES, generation_kwargs: dict = HUMAN_EVAL_GENERATION_KWARGS,
                        greedy_generation_kwargs: dict = HUMAN_EVAL_GREEDY_GENERATION_KWARGS):
    """Generate the HumanEvalInstruct completions for different temperatures with the model `model_name` and
    save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the HumanEval benchmark, by default HUMAN_EVAL_GENERATION_KWARGS
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the HumanEval benchmark, by default HUMAN_EVAL_GREEDY_GENERATION_KWARGS
    """

    # Load in 8 bits for bloom due to model size
    quantization = True if model_name == 'bloom-176B' else False

    model = engine.HFModel(model_name, quantization=quantization)
    stopping_patterns = None if (model.is_chat_model and prompt_template_mode in ['default', 'chat']) else stopping.CODE_STOP_PATTERNS
    folder = os.path.join(utils.RESULTS_FOLDER , f'HumanEvalInstruct_{prompt_template_mode}_{use_context}', 'completions', model_name)

    dataset = datasets.HumanEvalInstruct()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            task_id = sample['task_id']
            prompt = sample['instruction'] if use_context else sample['instruction'] + sample['context']
            context = sample['context'] if use_context else ''

            # GPT2 has only a context size of 1024, which can sometimes overflow with large `max_new_tokens`.
            if 'gpt2' in model_name:
                prompt_length = model.tokenizer.encode(prompt, return_tensors='pt').shape[-1]
                # Note that we need deepcopies to avoid changing the default values of the function inplace
                if prompt_length + generation_kwargs['max_new_tokens'] > 1024:
                    generation_kwargs = copy.deepcopy(generation_kwargs)
                    generation_kwargs['max_new_tokens'] = 1024 - prompt_length
                if prompt_length + greedy_generation_kwargs['max_new_tokens'] > 1024:
                    greedy_generation_kwargs = copy.deepcopy(greedy_generation_kwargs)
                    greedy_generation_kwargs['max_new_tokens'] = 1024 - prompt_length

            # In this case we use greedy decoding (the temperature parameters does not matter anymore
            # so we set it to the default which is 1)
            if temperature == 0:
                completions = [model(prompt, model_context=context, temperature=1., batch_size=1, stopping_patterns=stopping_patterns,
                                     prompt_template_mode=prompt_template_mode, **greedy_generation_kwargs)]
            # In this case we use top-p sampling
            else:
                completions = model(prompt, model_context=context, temperature=temperature, stopping_patterns=stopping_patterns,
                                    prompt_template_mode=prompt_template_mode, **generation_kwargs)

            # Save the model completions
            if model.is_chat_model and prompt_template_mode in ['default', 'chat']:
                true_completions = extract_completions(completions, sample)
                results = [{'task_id': task_id, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]
            else:
                results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HumanEval benchmark')
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
    
    args = parser.parse_args()
    big_models = args.big_models
    big_models_only = args.big_models_only
    special_only = args.special_only
    instruct = args.instruct
    mode = args.mode
    use_context = args.no_context

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    small_models = SMALL_MODELS_SPECIAL_PROMPT if special_only else SMALL_MODELS
    large_models = LARGE_MODELS_SPECIAL_PROMPT if special_only else LARGE_MODELS

    # Create the iterables to pass to the processing pool
    mode_iter = (mode,)*len(small_models)
    use_context_iter = (use_context,)*len(small_models)

    # target function
    target_func = human_eval_instruct if instruct else human_eval

    print(f'Launching computations with {num_gpus} gpus available.')

    if not big_models_only:
        args = (mode_iter, use_context_iter) if instruct else (mode_iter,)
        
        # Run all models that fit on a single gpu in parallel using all gpus
        # Use ProcessPoolExecutor() instead of mp.Pool() because it is slightly more convenient
        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn'),
                                initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
            _ = list(pool.map(target_func, small_models, *args, chunksize=1))

    if big_models or big_models_only:
        # Estimate number of gpus needed for each model
        model_footprints = []
        for model in large_models:
            quantization = model == 'bloom-176B'
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
            model_footprints.append(gpu_needed)

        args = (mode, use_context) if instruct else (mode,)

        utils.dispatch_jobs(large_models, model_footprints, num_gpus, target_func, args)


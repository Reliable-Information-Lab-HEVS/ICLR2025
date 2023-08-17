import os
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
import copy

import engine
from engine import stopping
from helpers import datasets
from helpers import utils

# TEMPERATURES = (0., 0.2, 0.4, 0.6, 0.8, 1.)
TEMPERATURES = (0.,)

# We need to set top_k to 0 to deactivate top-k sampling
HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': 0,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'seed': 1234,
    'truncate_prompt_from_output': True,
    'stopping_patterns': stopping.CODE_STOP_PATTERNS
}

HUMAN_EVAL_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'top_k': 0,
    'top_p': 1.,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
    'stopping_patterns': stopping.CODE_STOP_PATTERNS
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

LARGE_MODELS = (
    'gpt-neoX-20B',
    'opt-30B',
    'opt-66B',
    'llama-2-70B',
    'llama-2-70B-chat',
    'bloom-176B',
)


def human_eval(model_name: str, temperatures: tuple[int] = TEMPERATURES,
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

    model = engine.HFModel(model_name, quantization=quantization, gpu_rank=0)
    folder = os.path.join(utils.RESULTS_FOLDER , 'HumanEval_completions', model_name)

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
                completions = [model(prompt, temperature=1., batch_size=1, **greedy_generation_kwargs)]
            # In this case we use top-p sampling
            else:
                completions = model(prompt, temperature=temperature, **generation_kwargs)

            # Save the model completions
            results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


def human_eval_instruct(model_name: str, temperatures: tuple[int] = TEMPERATURES,
               generation_kwargs: dict = HUMAN_EVAL_GENERATION_KWARGS,
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

    model = engine.HFModel(model_name, quantization=quantization, gpu_rank=0)
    folder = os.path.join(utils.RESULTS_FOLDER , 'HumanEvalInstruct_completions', model_name)

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
            prompt = sample['instruction'] + sample['context']

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
                completions = [model(prompt, temperature=1., batch_size=1, **greedy_generation_kwargs)]
            # In this case we use top-p sampling
            else:
                completions = model(prompt, temperature=temperature, **generation_kwargs)

            # Save the model completions
            results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HumanEval benchmark')
    parser.add_argument('--gpus', type=int, default=8, help='The number of GPUs to use.')
    parser.add_argument('--big_models', type=str, default='False', choices=['False', 'True'],
                        help='Whether to run the benchmark on large models that do not fit on a single gpu.')
    parser.add_argument('--instruct', type=str, default='False', choices=['False', 'True'],
                        help='Run the HumanEvalInstruct benchmark.')
    
    args = parser.parse_args()
    num_gpus = args.gpus
    big_models = args.big_models == 'True'
    instruct = args.instruct == 'True'

    entry_point = human_eval if not instruct else human_eval_instruct

    # Run all models that fit on a single gpu in parallel using all gpus
    # Use ProcessPoolExecutor() instead of mp.Pool() because it is slightly more convenient
    # with mp.Pool(processes=num_gpus, initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        pool.map(entry_point, SMALL_MODELS, chunksize=1)

    if big_models:
        for model in LARGE_MODELS:
            entry_point(model)
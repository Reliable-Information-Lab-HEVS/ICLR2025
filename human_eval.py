import torch
import os
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
import copy

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
    'top_k': 0,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'seed': 1234,
    'truncate_prompt_from_output': True,
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
    'llama-2-70B',
    'llama-2-70B-chat',
    'bloom-176B',
)

LARGE_MODELS_SPECIAL_PROMPT = (
    'llama-2-70B-chat',
)


def dispatch_jobs(model_names, num_gpus, target_func, *func_args, **func_kwargs):
    """Run all jobs that need more than one gpu in parallel. Since the number of gpus needed by the models
    is variable, we cannot simply use a Pool of workers and map `target_func` to the Pool, or create processes and
    then ".join" them. To overcome this limitation, we use an infinite while loop that is refreshed by the main
    process every 10s. The dispatch of models to gpus is very naive: as soon as enough gpus are available to
    run the job that requires the less gpu, we launch it. Thus the gpu efficiency may not be the best possible.
    However, this would be extremely hard to improve on this simple strategy, especially since we do not know
    the runtime of each job.

    Parameters
    ----------
    model_names : _type_
        _description_
    num_gpus : _type_
        _description_
    """

    # TODO: set cuda devices then call target_func
    def target_func_on_gpu():
        pass

    model_names = list(model_names)
    model_footprints = []

    # Estimate number of gpus needed for each model
    for model in model_names:
        quantization = model == 'bloom-176B'
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
        model_footprints.append(gpu_needed)

    # Sort both lists according to gpu footprint
    sorting = sorted(zip(model_names, model_footprints), key=lambda x: x[1])
    model_names = [x for x, _ in sorting]
    model_footprints = [x for _, x in sorting]

    # Initialize the lists we will maintain
    available_gpus = [i for i in range(num_gpus)]
    processes = []
    associated_gpus = []

    while True:

        no_sleep = False

        if len(available_gpus) >= model_footprints[0]:

            no_sleep = True

            # Remove them from the list of models to process
            name = model_names.pop(0)
            footprint = model_footprints.pop(0)

            # Update gpu resources
            allocated_gpu = available_gpus[0:footprint]
            available_gpus = available_gpus[footprint:]

            p = mp.Process(target=target_func, args=func_args, kwargs=func_kwargs)
            p.start()

            # Add them to the list of running processes
            processes.append(p)
            associated_gpus.append(allocated_gpu)

        # Find the indices of the processes that are finished
        indices_to_remove = []
        for i, process in enumerate(processes):
            if not process.is_alive():
                indices_to_remove.append(i)

        # Update gpu resources
        released_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i in indices_to_remove]
        available_gpus += [gpu for gpus in released_gpus for gpu in gpus]
        # Remove processes which are done
        processes = [process for i, process in enumerate(processes) if i not in indices_to_remove]
        associated_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i not in indices_to_remove]

        # If we scheduled all jobs break from the infinite loop
        if len(model_names) == 0:
            break

        # Sleep for 10 seconds before restarting the loop and check if we have enough resources to launch
        # a new job
        if not no_sleep:
            time.sleep(10)

    # Sleep until all processes are finished (they have all been scheduled at this point)
    for process in processes:
        process.join()


def extract_completions(outputs: list[str], sample: dict, parser: CodeParser = PythonParser(),
                        stopping_patterns: tuple[str] = stopping.EXTENDED_CODE_STOP_PATTERNS) -> list[str]:
    
    code_outputs = stopping.parse_code_and_truncate(outputs, parser, stopping_patterns)

    completions = []
    for output in code_outputs:
        if output.startswith('def ' + sample['entry_point']):
            # Remove the function definition
            if '\n' in output:
                _, output = output.split('\n', 1)
                if '"""\n' in output:
                    _, completion = output.split('"""\n', 1)
                else:
                    completion = output
            else:
                completion = output
        else:
            completion = output

        completions.append(completion)

    return completions





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

    model = engine.HFModel(model_name, quantization=quantization, gpu_rank=0)
    stopping_patterns = None if (model.is_chat_model and prompt_template_mode in ['default', 'chat']) else stopping.CODE_STOP_PATTERNS
    folder = os.path.join(utils.RESULTS_FOLDER , f'HumanEval_completions_{prompt_template_mode}', model_name)

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

    model = engine.HFModel(model_name, quantization=quantization, gpu_rank=0)
    stopping_patterns = None if (model.is_chat_model and prompt_template_mode in ['default', 'chat']) else stopping.CODE_STOP_PATTERNS
    folder = os.path.join(utils.RESULTS_FOLDER , f'HumanEvalInstruct_completions_{prompt_template_mode}_{use_context}', model_name)

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
    parser.add_argument('--only_special', action='store_true',
                        help='If given, will only run the benchmark on models with a non-default prompt template.')
    parser.add_argument('--mode', type=str, default='default', choices=PROMPT_MODES,
                        help='The mode for the prompt template.')
    parser.add_argument('--instruct', action='store_true',
                        help='If given, run the HumanEvalInstruct benchmark.')
    parser.add_argument('--no_context', action='store_false',
                        help='If given, do NOT use the context in the HumanEvalInstruct benchmark.')
    
    args = parser.parse_args()
    big_models = args.big_models
    only_special = args.only_special
    instruct = args.instruct
    mode = args.mode
    use_context = args.no_context

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()
    print(num_gpus)

    small_models = SMALL_MODELS_SPECIAL_PROMPT if only_special else SMALL_MODELS
    large_models = LARGE_MODELS_SPECIAL_PROMPT if only_special else LARGE_MODELS

    # Create the iterables to pass to the processing pool
    mode_iter = (mode,)*len(small_models)
    use_context_iter = (use_context,)*len(small_models)

    # Run all models that fit on a single gpu in parallel using all gpus
    # Use ProcessPoolExecutor() instead of mp.Pool() because it is slightly more convenient
    # with mp.Pool(processes=num_gpus, initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context='spawn', initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        if instruct:
            _ = list(pool.map(human_eval_instruct, small_models, mode_iter, use_context_iter, chunksize=1))
        else:
            _ = list(pool.map(human_eval, small_models, mode_iter, chunksize=1))

    if big_models:
        for model in large_models:
            if instruct:
                human_eval_instruct(model, mode, use_context)
            else:
                human_eval(model, mode)


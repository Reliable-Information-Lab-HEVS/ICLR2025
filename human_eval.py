import os
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse
import time

import engine
from engine import stopping
from helpers import datasets
from helpers import utils

DATASET = datasets.HumanEval()

# We need to set top_k to 0 to deactivate top-k sampling
HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': 0,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'batch_size': 32,
    'seed': None,
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
    'seed': None,
    'truncate_prompt_from_output': True,
    'stopping_patterns': stopping.CODE_STOP_PATTERNS
}

TEMPERATURES = (0., 0.2, 0.4, 0.6, 0.8, 1.)

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
)

LARGE_MODELS = (
    'gpt-neoX-20B',
    'opt-30B',
    'opt-66B',
    'bloom-176B',
)


def find_rank_of_subprocess_inside_the_pool():
    """Find the rank of the current subprocess inside the pool that was launched either by 
    multiprocessing.Pool() or concurrent.futures.ProcessPoolExecutor().
    If called from the main process, return 0.
    Note that this is a bit hacky but work correctly because both methods provide the rank of the subprocesses
    inside the subprocesses name as 'SpawnPoolWorker-RANK' or 'SpawnProcess-RANK' respectively.
    """

    process = mp.current_process()

    if process.name == 'MainProcess':
        rank = 0
    elif isinstance(process, mp.context.SpawnProcess):
        # Provide rank starting at 0 instead of 1
        try:
            rank = int(process.name[-1]) - 1
        except ValueError:
            raise RuntimeError('Cannot retrieve the rank of the current subprocess.')
    else:
        raise RuntimeError('The type of the running process is unknown.')
        
    return rank



def human_eval(model_name: str, dataset: datasets.HumanEval = DATASET, temperatures: tuple[int] = TEMPERATURES,
               generation_kwargs: dict = HUMAN_EVAL_GENERATION_KWARGS,
               greedy_generation_kwargs: dict = HUMAN_EVAL_GREEDY_GENERATION_KWARGS):
    """Generate the HumanEval completions for different temperatures with the model `model_name` and
    save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    dataset : datasets.HumanEval, optional
        The HumanEval dataset, by default DATASET
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the HumanEval benchmark, by default HUMAN_EVAL_GENERATION_KWARGS
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the HumanEval benchmark, by default HUMAN_EVAL_GREEDY_GENERATION_KWARGS
    """

    # Load in 8 bits for bloom due to model size
    quantization = True if model_name == 'bloom-176B' else False

    gpu_rank = find_rank_of_subprocess_inside_the_pool()

    model = engine.HFModel(model_name, quantization=quantization, gpu_rank=gpu_rank)
    folder = os.path.join(utils.RESULTS_FOLDER , 'HumanEval_completions', model_name)

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')

        for sample in dataset:

            task_id = sample['task_id']
            prompt = sample['prompt']

            # In this case we use greedy decoding (the temperature parameters does not matter anymore
            # so we set it to the default which is 1)
            if temperature == 0:
                completions = [model(prompt, temperature=1., **greedy_generation_kwargs)]
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
    parser.add_argument('--big_models', type=str, default='True', choices=['False', 'True'],
                        help='Whether to run the benchmark on large models that do not fit on a single gpu.')
    
    args = parser.parse_args()
    num_gpus = args.gpus
    big_models = args.big_models == 'True'

    # Run all models that fit on a single gpu in parallel using all gpus
    # Use ProcessPoolExecutor() instead of mp.Pool() because it is slightly more convenient
    # with mp.Pool(processes=num_gpus, initializer=utils.set_all_seeds, initargs=(1234,)) as pool:
    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.set_all_seeds, initargs=(1234,)) as pool:
        pool.map(human_eval, SMALL_MODELS, chunksize=1)

    if big_models:
        for model in LARGE_MODELS:
            human_eval(model)
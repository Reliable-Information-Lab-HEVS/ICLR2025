import os
import gc
import argparse
import time

from TextWiz import textwiz
from helpers import datasets
from helpers import utils
from helpers import aatk

TEMPERATURES = (0.2,)

# We need to set top_k to 0 to deactivate top-k sampling
AATK_GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 25,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

AATK_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}



def aatk_english_chatgpt_benchmark(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                           temperatures: tuple[int] = TEMPERATURES,
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

    prompt_template_mode = 'generation' if model_name == 'codegen25-7B-instruct' else 'chat'

    if not textwiz.is_chat_model(model_name):
        raise ValueError('Cannot run AATKEnglish benchmark on non-chat model.')

    print(f'{utils.get_time()}  Starting with model {model_name}')

    model = textwiz.HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    folder = aatk.get_folder('AATK_english_chatGPT', model_name, model.dtype_category())

    dataset = datasets.AATKEnglishChatGPT()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            id = sample['id']
            prompts = [sample['intent']] + sample['intent_variations']
            prompt_names = ['original'] + [f'variation {i}' for i in range(len(prompts)-1)]

            for prompt, prompt_name in zip(prompts, prompt_names):

                # In this case we use greedy decoding
                if temperature == 0:
                    completions = [model(prompt, batch_size=1, prompt_template_mode=prompt_template_mode,
                                         stopping_patterns=None, **greedy_generation_kwargs)]
                # In this case we use top-p sampling
                else:
                    completions = model(prompt, temperature=temperature, prompt_template_mode=prompt_template_mode,
                                        stopping_patterns=None, **generation_kwargs)
                    
                # Remove trailing whitespaces
                completions = [completion.rstrip() for completion in completions]
            
                results = [{'id': id, 'prompt': prompt, 'prompt_id': prompt_name, 'completion': completion} for completion in completions]
                utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'{utils.get_time()}  Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()



def aatk_english_zephyr_benchmark(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                                  temperatures: tuple[int] = TEMPERATURES,
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

    prompt_template_mode = 'generation' if model_name == 'codegen25-7B-instruct' else 'chat'

    if not textwiz.is_chat_model(model_name):
        raise ValueError('Cannot run AATKEnglish benchmark on non-chat model.')

    print(f'{utils.get_time()}  Starting with model {model_name}')

    model = textwiz.HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    folder = aatk.get_folder('AATK_english_zephyr', model_name, model.dtype_category())

    dataset = datasets.AATKEnglishZephyr()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            id = sample['id']
            prompts = [sample['intent']['prompt']] + [x['prompt'] for x in sample['intent_variations']]
            perplexities = [sample['intent']['perplexity']] + [x['perplexity'] for x in sample['intent_variations']]
            prompt_names = ['original'] + [f'variation {i}' for i in range(len(prompts)-1)]

            for prompt, prompt_name, ppl in zip(prompts, prompt_names, perplexities):

                # In this case we use greedy decoding
                if temperature == 0:
                    completions = [model(prompt, batch_size=1, prompt_template_mode=prompt_template_mode,
                                         stopping_patterns=None, **greedy_generation_kwargs)]
                # In this case we use top-p sampling
                else:
                    completions = model(prompt, temperature=temperature, prompt_template_mode=prompt_template_mode,
                                        stopping_patterns=None, **generation_kwargs)
                    
                # Remove trailing whitespaces
                completions = [completion.rstrip() for completion in completions]
            
                results = [{'id': id, 'prompt': prompt, 'prompt_id': prompt_name, 'perplexity': ppl, 'completion': completion} for completion in completions]
                utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'{utils.get_time()}  Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AATK english benchmark')
    parser.add_argument('model', type=str,
                        help='The model to run.')
    parser.add_argument('--version', type=int, choices=[1,2], default=1,
                        help='Version of the AATK english benchmark (1 for prompt reformulation with ChatGPT, 2 for Zephyr)')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    
    args = parser.parse_args()
    version = args.version
    model = args.model
    int8 = args.int8
    int4 = args.int4

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    if version == 1:
        aatk_english_chatgpt_benchmark(model, int8, int4)
    else:
        aatk_english_zephyr_benchmark(model, int8, int4)


from TextWiz.textwiz import HFModel, loader
from helpers import utils

import argparse
import warnings
import json
import gc
import os
import re

GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': True,
    'temperature': 0.2,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 25,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': False,
    'num_return_sequences': 1,
    'batch_size': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}


class BadFormatException(Exception):
    """Custom Exception to catch if format of model is incorrect.
    """
    pass


def parse_output(output: str, N: int) -> list[str]:
    """Parse the output of the model asked to create prompt variation, and raise `BadFormatException` if the
    format is incorrect.

    Parameters
    ----------
    output : str
        Output of the model asked to generate the prompt variations.
    N : int
        Number of prompt variations.

    Returns
    -------
    list[str]
        The `N` prompts if the parsing is successful.
    """

    # Pattern that matches the enumeration format. We add the capturing group to also keep the separators
    prompts = re.split(r'((?:^|\n)[0-9]+\. )', output)

    # The split pattern usually creates a first empty string as it matches at the immediate beginning of output
    if prompts[0] == '':
        prompts.pop(0)

    # Rejoin each separator and what is after it, and strip whitespaces
    prompts = [''.join((prompts[i], prompts[i+1])).strip() for i in range(0, len(prompts), 2)]

    # format error
    if len(prompts) != N:
        raise BadFormatException('Cannot find `N` variations of the prompt')
    
    # Repetition
    if len(set(prompts)) != len(prompts):
        raise BadFormatException('The same prompt was repeated')
    
    # Check that the enumeration numbers (the separators in `split`) are correctly ordered
    formatted_prompts = []
    for i, prompt in enumerate(prompts, 1):

        # This is a format error
        if not prompt.startswith(f'{i}. '):
            raise BadFormatException('The listing format is incorrect')
        
        formatted_prompts.append(prompt.replace(f'{i}. ', '', 1))

    return formatted_prompts


def create_variations(model: HFModel, original_prompt: str, N: int = 10, recursion_depth: int = 10) -> list[str]:
    """Use `model` to create `N` other formulations of `original_prompt`. This function will retry the generation
    of the prompts `recursion_depth` times if the parsing of the output is unsuccessful before abandoning.

    Parameters
    ----------
    model : HFModel
        The model to use.
    original_prompt : str
        The original prompt to use.
    N : int, optional
        How many new prompts to generate. By default 10.
    recursion_depth : int, optional
        How many retries we allow before abandoning the creation of the prompts.

    Returns
    -------
    list[str]
        The prompt variations.
    """

    if not isinstance(N, int):
        raise RuntimeError('`N` must be an int.')
    
    prompt = f'Give me {N} reformulations (without repeating yourself) of the following instruction: "{original_prompt}"'

    # Try greedy decoding first (use some repetition penalty to create more diverse prompts)
    out = model(prompt, max_new_tokens=4096, do_sample=False, batch_size=1, stopping_patterns=[f'\n{N+1}. '],
                repetition_penalty=1.1)
    try:
        prompts = parse_output(out, N)
        return prompts
    except BadFormatException:
        pass

    # Greedy decoding failed to output a correct format -> try stochastic sample of the new tokens
    recursion_count = 0
    # We try to generate 10 times the new prompts, and abandon afterwards
    while recursion_count < recursion_depth:

        recursion_count += 1
        # We use a larger repetition_penalty because its effect gets smoothed by the low temperature
        out = model(prompt, max_new_tokens=4096, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1,
                    stopping_patterns=[f'\n{N+1}. '], repetition_penalty=1.15, seed=123)
        try:
            prompts = parse_output(out, N)
            return prompts
        except BadFormatException:
            pass
    
    raise BadFormatException(f'Could not correctly generate {N} variations after {recursion_depth} tries.')


def main(main_model: str, sub_model: str, input_file: str, output_file: str, existing_variations: bool,
         N: int, generation_kwargs: dict):
    """Create `N` different versions of the prompts in `input_file`, and then use `main_model` to generate
    text based on those prompts and `generation_kwargs`. If `existing_variations` is True, ìnput_file` is assumed
    to already contain the prompts variations and they are not re-generated. Save results to `output_file`.

    Parameters
    ----------
    main_model : str
        The model to use to generate text based on the prompts.
    sub_model : str
        The model to use to generate the prompt variations.
    input_file : str
        Path to the file containing the prompts.
    output_file : str
        Where to save the results.
    existing_variations : bool
        If `True`, the prompts variations are not recomputed.
    N : int
        How many prompts to generate.
    generation_kwargs : dict
        Generation parameters used by `main_model`.
    """

    # Make sure output file will be writable
    dirname, basename = os.path.split(output_file)
    if basename == '':
        raise ValueError('The output file cannot end by directory separator.')
    if not basename.endswith('.jsonl'):
        warnings.warn('The output file extension should be `.jsonl`. Adding the extension to given filename.')
        basename += '.jsonl'
    if dirname != '':
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    output_file = os.path.join(dirname, basename)

    # Load the prompts and create the variations
    if not existing_variations:
        prompts = utils.load_txt(input_file, separator='\n\n')
        prompts = [prompt.strip() for prompt in prompts if prompt.strip() != '']

        sub_model = HFModel(sub_model)

        prompt_bank = []
        for prompt in prompts:
            # Try to create the prompt variations
            try:
                variations = create_variations(sub_model, prompt, N)
            except BadFormatException:
                warnings.warn(f'Could not create {N} variations of the following prompt (ignoring it):\n{prompt}')
                continue

            # Compute the perplexity of each prompt
            original_perplexity = sub_model.perplexity(prompt)
            perplexities = [sub_model.perplexity(x) for x in variations]
            # Add to the prompt bank
            prompt_bank.append({'original_prompt': prompt, 'prompt': prompt, 'prompt_perplexity': original_perplexity})
            prompt_bank.extend([{'original_prompt': prompt, 'prompt': x, 'prompt_perplexity': y} for x, y in zip(variations, perplexities)])

        # Save the generated prompts
        basename, _ = os.path.splitext(input_file)
        name = basename + '_extended.jsonl'
        utils.save_jsonl(prompt_bank, name)

        del sub_model
        gc.collect()

    else:
        prompt_bank = utils.load_jsonl(input_file)

    # Perform inference on all the prompts
    model = HFModel(main_model)
    for sample in prompt_bank:
        prompt = sample['prompt']
        outputs = model(prompt, **generation_kwargs)
        sample['output'] = outputs
        with open(output_file, 'a') as fp:
            fp.write(json.dumps(sample) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text generation')
    parser.add_argument('model', type=str, choices=loader.ALLOWED_MODELS,
                        help='The model to use for inference.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the file containing the prompts.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='A path to save the output file.')
    parser.add_argument('--existing_variations', action='store_true',
                        help='If given, `input_file` is assumed to already contain the prompt variations.')
    # parser.add_argument('--sub_model', type=str, default='zephyr-7B-beta',
    #                     help='The model to use to create prompt variations.')
    parser.add_argument('--greedy', action='store_true',
                        help='If given, will use greedy decoding for inference (i.e. only one completion per prompt).')
    parser.add_argument('--N', type=int, default=10,
                        help='How many variations of each prompt to create.')
    
    args = parser.parse_args()
    main_model = args.model
    # sub_model = args.sub_model
    sub_model = 'zephyr-7B-beta'
    input_file = args.input_file
    output_file = args.output_file
    generation_kwargs = GREEDY_GENERATION_KWARGS if args.greedy else GENERATION_KWARGS
    existing_variations = args.existing_variations
    N = args.N

    main(main_model=main_model, sub_model=sub_model, input_file=input_file, output_file=output_file,
         existing_variations=existing_variations, N=N, generation_kwargs=generation_kwargs)


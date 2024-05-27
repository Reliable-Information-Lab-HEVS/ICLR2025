import argparse
import warnings
import os
import re
from tqdm import tqdm

from TextWiz.textwiz import HFCausalModel, loader
from helpers import utils, datasets


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

    print(output)

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


def create_variations(model: HFCausalModel, original_prompt: str, N: int = 10, recursion_depth: int = 10) -> list[str]:
    """Use `model` to create `N` other formulations of `original_prompt`. This function will retry the generation
    of the prompts `recursion_depth` times if the parsing of the output is unsuccessful before abandoning.

    Parameters
    ----------
    model : HFCausalModel
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


def map_output_to_AATK_format(output_bank: list[dict]) -> list[dict]:
    """Map created variations to AATK format (mapping by id)"""

    AATK_samples = datasets.AATK.samples_by_id()
    out = []
    for sample in output_bank:
        aatk_sample = AATK_samples[sample['id']]
        aatk_sample['intent'] = sample['original_prompt']
        aatk_sample['intent_variations'] = sample['prompt_variations']
        out.append(aatk_sample)

    return out


def main(model: str, input_file: str, output_file: str, N: int, map_to_AATK_format: bool):
    """Create `N` different versions of the prompts in `input_file`, and then use `main_model` to generate
    text based on those prompts and `generation_kwargs`. If `existing_variations` is True, ìnput_file` is assumed
    to already contain the prompts variations and they are not re-generated. Save results to `output_file`.

    Parameters
    ----------
    model : str
        The model to use to generate the prompt variations.
    input_file : str
        Path to the file containing the prompts.
    output_file : str
        Where to save the results.
    N : int
        How many prompts to generate.
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

    # Create variations
    samples = utils.load_jsonl(input_file)
    assert all(['prompt' in sample.keys() for sample in samples]), 'Input file format is incorrect'

    model = HFCausalModel(model)

    output_bank = []
    for sample in tqdm(samples):
        prompt = sample['prompt']
        # Try to create the prompt variations
        try:
            variations = create_variations(model, prompt, N)
        except BadFormatException:
            warnings.warn(f'Could not create {N} variations of the following prompt (ignoring it):\n{prompt}')
            continue

        output_sample = {k:v for k,v in sample.items() if k != 'prompt'}
        output_sample['original_prompt'] = prompt
        output_sample['prompt_variations'] = variations
        # Add to the output bank
        output_bank.append(output_sample)

        # Save the generated prompts
        utils.save_jsonl(output_bank, output_file)

    if map_to_AATK_format:
        # Save the generated prompts
        output_bank = map_output_to_AATK_format(output_bank)
        utils.save_jsonl(output_bank, output_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text generation')
    parser.add_argument('model', type=str, choices=loader.ALLOWED_CAUSAL_MODELS,
                        help='The model to use for the reformulations.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the file containing the prompts.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='A path to save the output file.')
    parser.add_argument('--N', type=int, default=10,
                        help='How many variations of each prompt to create.')
    parser.add_argument('--map_to_AATK_format', action='store_true',
                        help='If given, try to map to the AATK dataset format.')
    
    args = parser.parse_args()
    model = args.model
    input_file = args.input_file
    output_file = args.output_file
    N = args.N
    map_to_AATK_format = args.map_to_AATK_format

    main(model=model, input_file=input_file, output_file=output_file, N=N,
         map_to_AATK_format=map_to_AATK_format)


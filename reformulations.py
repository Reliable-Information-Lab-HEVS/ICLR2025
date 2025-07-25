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

    # Pattern that matches the enumeration format. We add the capturing group to also keep the separators
    prompts = re.split(r'((?:^|\n)[0-9]+\. )', output)

    # First item in the list is usually "Here are the reformulations..."
    if not prompts[0].startswith('1.'):
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
    # We try to generate 10 times the new prompts, and abandon afterwards
    # We use a larger repetition_penalty because its effect gets smoothed by the low temperature
    outputs = model(prompt, max_new_tokens=4096, do_sample=True, temperature=0.4, top_p=0.9, top_k=30,
                    num_return_sequences=10, stopping_patterns=[f'\n{N+1}. '], repetition_penalty=1.15, seed=123)
    
    for out in outputs:
        try:
            prompts = parse_output(out, N)
            return prompts
        except BadFormatException:
            pass
    
    raise BadFormatException(f'Could not correctly generate {N} variations after {recursion_depth} tries.')


def main(model: str, input_file: str, output_file: str, N: int, original_key: str, new_key: str = 'prompt_variations'):
    """Create `N` different versions of the prompts in `input_file`. Save results to `output_file`.

    Parameters
    ----------
    model : str
        The model to use to generate the prompt variations.
    input_file : str
        Path to the file containing the prompts. It should be a `jsonl` file containing one json serialized object
        per new line.
    output_file : str
        Where to save the results.
    N : int
        How many prompts to generate.
    original_key : str
        Name of the key holding the prompts to reformulate in each dictionary in `input_file`.
    new_key : str, optional
        Name of the dictionary key used to save the prompt variations in the new dictionaries. By default 'prompt_variations'.
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
    assert all([original_key in sample.keys() for sample in samples]), 'The `original_key` is not present in all samples'

    print("Running model: ", model)
    model = HFCausalModel(model)

    output_bank = []
    for sample in tqdm(samples):
        prompt = sample[original_key]
        # Try to create the prompt variations
        try:
            variations = create_variations(model, prompt, N)
        except BadFormatException:
            warnings.warn(f'Could not create {N} variations of the following prompt (ignoring it):\n{prompt}')
            continue

        output_sample = sample.copy()
        output_sample[new_key] = variations
        # Add to the output bank
        output_bank.append(output_sample)

        # Save the generated prompts
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
    parser.add_argument('--original_key', type=str, required=True,
                        help='The dictionary key containing the prompts to reformulate in `input_file`.')
    parser.add_argument('--new_key', type=str, default='prompt_variations',
                        help='The dictionary key in which to save the prompt reformulations in `output_file`.')
    
    args = parser.parse_args()
    model = args.model
    input_file = args.input_file
    output_file = args.output_file
    N = args.N
    original_key = args.original_key
    new_key = args.new_key

    main(model=model, input_file=input_file, output_file=output_file, N=N, original_key=original_key, new_key=new_key)


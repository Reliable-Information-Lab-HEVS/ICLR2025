from TextWiz.textwiz import HFModel
from helpers import utils

import argparse
import json
import gc
import os

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


def create_variations(model: HFModel, original_prompt: str, N: int = 10) -> list[str]:
    """Use `model` to create `N` other formulations of `original_prompt`.

    Parameters
    ----------
    model : HFModel
        The model to use.
    original_prompt : str
        The original prompt to use.
    N : int, optional
        How many new prompts to generate. By default 10.

    Returns
    -------
    list[str]
        The prompt variations.
    """

    if not isinstance(N, int):
        raise RuntimeError('`N` must be an int.')

    prompt = f'Give me {N} reformulations of this: "{original_prompt}"'
    out = model(prompt, max_new_tokens=2048, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1)

    return out


def main(main_model: str, sub_model: str, input_file: str, output_file: str, N: int, generation_kwargs: dict):
    """Create `N` different versions of the prompts in `input_file`, and then use `main_model` to generate
    text based on those prompts and `generation_kwargs`. Save results to `output_file`.

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
        basename += '.jsonl'
    if dirname != '':
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    output_file = os.path.join(dirname, basename)

    # Load the prompts and create the variations
    prompts = utils.load_txt(input_file, separator='\n\n')

    sub_model = HFModel(sub_model)

    prompt_bank = []
    for prompt in prompts:
        prompt_bank.append({'original_prompt': prompt, 'prompt': prompt})
        variations = create_variations(sub_model, prompt, N)
        prompt_bank.extend([{'original_prompt': prompt, 'prompt': x} for x in variations])
    
    del sub_model
    gc.collect()

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
    parser.add_argument('model', type=str,
                        help='The model to use for inference.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the file containing the prompts.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='A path to save the output file.')
    parser.add_argument('--sub_model', type=str, default='zephyr-7B-beta',
                        help='The model to use to create prompt variations.')
    parser.add_argument('--greedy', action='store_true',
                        help='If given, will use greedy decoding for inference (i.e. only one completion per prompt).')
    parser.add_argument('--N', type=int, default=10,
                        help='How many variations of each prompt to create.')
    
    args = parser.parse_args()
    main_model = args.model
    sub_model = args.sub_model
    input_file = args.input_file
    output_file = args.output_file
    generation_kwargs = GREEDY_GENERATION_KWARGS if args.greedy else GENERATION_KWARGS
    N = args.N

    main(main_model=main_model, sub_model=sub_model, input_file=input_file, output_file=output_file,
         N=N, generation_kwargs=generation_kwargs)


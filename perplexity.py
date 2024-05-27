import argparse
from tqdm import tqdm

from TextWiz.textwiz import HFCausalModel, loader
from helpers import utils


def main(model: str, input_file: str, output_file: str, AATK_format: bool):
   
    # Load dataset
    samples = utils.load_jsonl(input_file)

    # Append to the output file if it already exist
    try:
        output = utils.load_json(output_file)
        append_to_file = True
    except FileNotFoundError:
        output = {}
        append_to_file = False

    model = HFCausalModel(model)

    for sample in tqdm(samples):

        original_key = 'intent' if AATK_format else 'original_prompt'
        variations_key = 'intent_variations' if AATK_format else 'prompt_variations'
        all_prompts = [sample[original_key]] + sample[variations_key]
        
        # Compute perplexities
        for prompt in all_prompts:
            if not append_to_file:
                output[prompt] = {}
            output[prompt][model.model_name] = model.perplexity(prompt)

    utils.save_json(output, output_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text generation')
    parser.add_argument('model', type=str, choices=loader.ALLOWED_CAUSAL_MODELS,
                        help='The model to use to compute the perplexities.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the file containing the prompts.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='A path to save the output file.')
    parser.add_argument('--AATK_format', action='store_true',
                        help='If given, assume that the file is in the AATK dataset format')
    
    args = parser.parse_args()
    model = args.model
    input_file = args.input_file
    output_file = args.output_file
    AATK_format = args.AATK_format

    main(model=model, input_file=input_file, output_file=output_file, AATK_format=AATK_format)


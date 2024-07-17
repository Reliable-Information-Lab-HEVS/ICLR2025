import os
import argparse
import sys
from tqdm import tqdm

from TextWiz import textwiz
from helpers import cybersec, datasets, utils

TEMPERATURES = (0.2,)

CYBER_SEC_EVAL_GENERATION_KWARGS = {
    'prompt_template_mode': 'generation',
    'max_new_tokens': 2048,
    'min_new_tokens': 0,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 10,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

CYBER_SEC_EVAL_GREEDY_GENERATION_KWARGS = {
    'prompt_template_mode': 'generation',
    'max_new_tokens': 2048,
    'min_new_tokens': 0,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}



DATASET_NAME_MAPPING = {
    'original': 'CyberSecEval_instruct',
    'llama3': 'CyberSecEval_instruct_llama3',
}

DATASET_MAPPING = {
    'CyberSecEval_instruct': datasets.CyberSecEvalInstruct,
    'CyberSecEval_instruct_llama3': datasets.CyberSecEvalInstructLlama3,
}


def cyber_sec_eval_instruct_benchmark(model_name: str, dataset: str, quantization_8bits: bool = False,
                                      quantization_4bits: bool = False, temperatures: tuple[int] = TEMPERATURES,
                                      generation_kwargs: dict = CYBER_SEC_EVAL_GENERATION_KWARGS,
                                      greedy_generation_kwargs: dict = CYBER_SEC_EVAL_GREEDY_GENERATION_KWARGS):
    """Generate the CyberSecEval completions for different temperatures with the model `model_name` and
    save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    dataset : str
        Which version of the CyberSecEval benchmark to run.
    quantization_8bits : bool
        Whether to use 8 bits quantization, by default False.
    quantization_4bits : bool
        Whether to use 4 bits quantization, by default False.
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the CyberSecEval benchmark, by default CYBER_SEC_EVAL_GENERATION_KWARGS.
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the CyberSecEval benchmark, by default CYBER_SEC_EVAL_GREEDY_GENERATION_KWARGS.
    """

    if not textwiz.is_chat_model(model_name):
        raise ValueError('Cannot run cyberSecEvalInstruct benchmark on non-chat model.')
    
    dtype_name = textwiz.dtype_category(model_name, quantization_4bits, quantization_8bits)
    folder = cybersec.get_folder(dataset, model_name, dtype_name)

    # Load a proper Dataset object
    dataset = DATASET_MAPPING[dataset]()

    # Check if results already exist
    valid_temperatures = list(temperatures)
    for temperature in temperatures:
        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        if os.path.exists(filename):
            if len(utils.load_jsonl(filename)) == len(dataset):
                valid_temperatures.remove(temperature)
    # In this case immediately return
    if len(valid_temperatures) == 0:
        print(f'The benchmark for {model_name} already exists.')
        return

    # Load model
    model = textwiz.HFCausalModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    for temperature in valid_temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Check potential partial results
        if os.path.exists(filename):
            initial_results = utils.load_jsonl(filename)
            # Remove completions keys for later easy matching in `sample_is_done`
            for initial_sample in initial_results:
                initial_sample.pop('original_completions')
                if 'reformulation_completions' in initial_sample:
                    initial_sample.pop('reformulation_completions')
        else:
            initial_results = None

        for sample in tqdm(dataset, desc=model_name, file=sys.stdout):

            # Check if results already exist for given sample, and continue to next sample if True
            if sample_is_done(sample, initial_results):
                continue

            prompts = [sample['test_case_prompt'].strip()]
            if 'test_case_prompt_reformulations' in sample:
                prompts += [x.strip() for x in sample['test_case_prompt_reformulations']]

            results = sample.copy()
            # List of list containing all completions for each prompt variation
            all_completions = []
            for prompt in prompts:

                # In this case we use greedy decoding
                if temperature == 0:
                    completions = [model(prompt, batch_size=1, stopping_patterns=None, **greedy_generation_kwargs)]
                # In this case we use top-p sampling
                else:
                    completions = model(prompt, temperature=temperature, stopping_patterns=None, **generation_kwargs)
                    
                # Remove trailing whitespaces
                completions = [completion.rstrip() for completion in completions]
                # Add them to the list of all results
                all_completions.append(completions)
            
            # Add completions to result dict
            results['original_completions'] = all_completions[0]
            if len(prompts) > 1:
                results['reformulation_completions'] = all_completions[1:]
            # Save to file
            utils.save_jsonl(results, filename, append=True)


def sample_is_done(sample: dict, initial_results: list[dict] | None) -> bool:
    """Check if a given `sample` from the dataset already exists in `initial_results`."""
    if initial_results is None:
        return False
    else:
        return any(sample == initial_sample for initial_sample in initial_results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CyberSecEval instruct benchmark')
    parser.add_argument('model', type=str, help='The model to run.')
    parser.add_argument('--dataset', type=str, required=True, choices=('original', 'llama3'),
                        help='The dataset version to run.')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    
    args = parser.parse_args()
    model = args.model
    dataset = DATASET_NAME_MAPPING[args.dataset]
    int8 = args.int8
    int4 = args.int4

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    cyber_sec_eval_instruct_benchmark(model, dataset, int8, int4)


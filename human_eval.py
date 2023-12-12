import os
import gc
import argparse
import time
import copy
import re

from TextWiz.textwiz import HFModel, stopping
from TextWiz.textwiz.prompt_template import PROMPT_MODES
from TextWiz.textwiz.code_parser import CodeParser, PythonParser
from helpers import datasets
from helpers import utils
from helpers import humaneval

# TEMPERATURES = (0., 0.2, 0.4, 0.6, 0.8, 1.)
TEMPERATURES = (0.,)
# TEMPERATURES = (0.8,)

# We need to set top_k to 0 to deactivate top-k sampling
HUMAN_EVAL_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

HUMAN_EVAL_GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 512,
    'min_new_tokens': 5,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}



def extract_completions(outputs: list[str], sample: dict, parser: CodeParser = PythonParser(),
                        stopping_patterns: tuple[str] = stopping.EXTENDED_CODE_STOP_PATTERNS) -> list[str]:
    
    code_outputs = [parser(sequence) for sequence in outputs]

    completions = []
    for output in code_outputs:

        # Check if the model repeated the function definition
        regex = r'def ' + re.escape(sample['entry_point']) + r'[^\n]*\n(.*)$'
        code_after_func_def = re.search(regex, output, re.DOTALL)

        if code_after_func_def:
            output = code_after_func_def.group(1)
            output = stopping.post_process_stopping_patterns([output], stopping_patterns)[0]
            # Remove the function definition
            if re.search(r'"""(.*?)"""', output, re.DOTALL):
                _, _, completion = output.split('"""', 2)
            else:
                completion = output
        else:
            completion = stopping.post_process_stopping_patterns([output], stopping_patterns)[0]

        completions.append(completion)

    return completions



def human_eval(model_name: str, prompt_template_mode: str, quantization_8bits: bool = False,
               quantization_4bits: bool = False, temperatures: tuple[int] = TEMPERATURES,
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

    print(f'{utils.get_time()}  Starting with model {model_name}')

    # Override quantization for bloom because it's too big
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        model = HFModel(model_name, quantization_8bits=True)
    else:
        model = HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    # Define stopping type
    if model.is_chat_model() and (prompt_template_mode == 'default' or prompt_template_mode == 'chat'):
        stopping_patterns = None
    else:
        stopping_patterns = stopping.StoppingType.PYTHON_HUMAN_EVAL

    folder = humaneval.get_folder('HumanEval', prompt_template_mode, model_name, model.dtype_category())

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
            prompt = sample['prompt'].strip()

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
            if model.is_chat_model() and (prompt_template_mode == 'default' or prompt_template_mode == 'chat'):
                true_completions = extract_completions(completions, sample)
                results = [{'task_id': task_id, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]
            else:
                results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'{utils.get_time()}  Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()



def human_eval_instruct(model_name: str, prompt_template_mode: str, use_context: bool, quantization_8bits: bool = False,
                        quantization_4bits: bool = False, temperatures: tuple[int] = TEMPERATURES,
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

    print(f'{utils.get_time()}  Starting with model {model_name}')

    # Override quantization for bloom because it's too big
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        model = HFModel(model_name, quantization_8bits=True)
    else:
        model = HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    # Define stopping type
    if model.is_chat_model():
        stopping_patterns = None
    else:
        stopping_patterns = stopping.StoppingType.PYTHON_HUMAN_EVAL

    folder = humaneval.get_folder('HumanEvalInstruct', prompt_template_mode, model_name, model.dtype_category(),
                                  use_context=use_context)

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
            prompt = sample['instruction'].strip() if use_context else sample['instruction'].strip() + '\n' + sample['context'].strip()
            context = sample['context'].strip() if use_context else ''

            # Add the newline character in case use_context is True but the model cannot actually handle context
            # correctly
            if use_context and model.prompt_template.default_mode == 'generation':
                prompt += '\n'

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
            if model.is_chat_model():
                true_completions = extract_completions(completions, sample)
                results = [{'task_id': task_id, 'model_output': x, 'completion': y} for x, y in zip(completions, true_completions)]
            else:
                results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'{utils.get_time()}  Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()



def human_eval_multiple(model_name: str, language: str, quantization_8bits: bool = False,
                        quantization_4bits: bool = False, temperatures: tuple[int] = TEMPERATURES,
                        generation_kwargs: dict = HUMAN_EVAL_GENERATION_KWARGS,
                        greedy_generation_kwargs: dict = HUMAN_EVAL_GREEDY_GENERATION_KWARGS):
    """Generate the MultiPL-E HumanEval completions for `language` fordifferent temperatures with the model
    `model_name` and save the results.

    Parameters
    ----------
    model_name : str
        The model name.
    language : str
        The programming language.
    temperatures : tuple[int], optional
        The different temperaturs to use to generate the completions, by default TEMPERATURES
    generation_kwargs : dict, optional
        The argument for generation used in the HumanEval benchmark, by default HUMAN_EVAL_GENERATION_KWARGS
    greedy_generation_kwargs : dict, optional
        The argument for greedy generation used in the HumanEval benchmark, by default HUMAN_EVAL_GREEDY_GENERATION_KWARGS
    """

    if language not in datasets.MULTIPLE_LANGUAGE_MAPPING.keys():
        raise ValueError(f'The language must be one of {*datasets.MULTIPLE_LANGUAGE_MAPPING.keys(),}.')

    print(f'{utils.get_time()}  Starting with model {model_name}')

    # Override quantization for bloom because it's too big
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        model = HFModel(model_name, quantization_8bits=True)
    else:
        model = HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    dataset_name = datasets.MULTIPLE_LANGUAGE_MAPPING[language]

    folder = humaneval.get_folder(dataset_name, 'generation', model_name, model.dtype_category())

    dataset = datasets.HUMANEVAL_DATASETS_MAPPING[dataset_name]()

    t0 = time.time()

    for temperature in temperatures:

        filename = os.path.join(folder, f'temperature_{temperature}.jsonl')
        # Delete the file if it already exist for some reason (e.g. a previous run that dit not end correctly)
        # because in this case we do not want to append to the file
        if os.path.exists(filename):
            os.remove(filename)

        for sample in dataset:

            task_id = sample['task_id']
            prompt = sample['prompt'].strip()
            stopping_patterns = sample['stop_tokens']

            # In this case we use greedy decoding (the temperature parameters does not matter anymore
            # so we set it to the default which is 1)
            if temperature == 0:
                completions = [model(prompt, temperature=1., batch_size=1, stopping_patterns=stopping_patterns,
                                     prompt_template_mode='generation', **greedy_generation_kwargs)]
            # In this case we use top-p sampling
            else:
                completions = model(prompt, temperature=temperature, stopping_patterns=stopping_patterns,
                                    prompt_template_mode='generation', **generation_kwargs)

            # Save the model completions
            results = [{'task_id': task_id, 'completion': completion} for completion in completions]
            utils.save_jsonl(results, filename, append=True)

    dt = time.time() - t0

    print(f'{utils.get_time()}  Done with model {model_name} in {dt/3600:.2f}h!')
    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HumanEval benchmark')
    parser.add_argument('model', type=str,
                        help='The model to run.')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--mode', type=str, default='generation', choices=PROMPT_MODES,
                        help='The mode for the prompt template.')
    parser.add_argument('--instruct', action='store_true',
                        help='If given, run the HumanEvalInstruct benchmark.')
    parser.add_argument('--no_context', action='store_false',
                        help='If given, do NOT use the context in the HumanEvalInstruct benchmark.')
    parser.add_argument('--language', type=str, default='py',
                        help='If given, run the corresponding benchmark.')
    
    args = parser.parse_args()
    model = args.model
    int8 = args.int8
    int4 = args.int4
    instruct = args.instruct
    mode = args.mode
    use_context = args.no_context
    language = args.language

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')
    
    if instruct and language != 'py':
        raise ValueError('instruct and language options are mutually exclusive.')
    
    if language not in datasets.MULTIPLE_LANGUAGE_MAPPING.keys() and language != 'py':
        raise ValueError(f'The language must be one of {*datasets.MULTIPLE_LANGUAGE_MAPPING.keys(),}.')
    
    # target function
    if instruct:
        target_func = human_eval_instruct
        args = (model, mode, use_context, int8, int4)
    elif language != 'py':
        target_func = human_eval_multiple
        args = (model, language, int8, int4)
    else:
        target_func = human_eval
        args = (model, mode, int8, int4)

    target_func(*args)


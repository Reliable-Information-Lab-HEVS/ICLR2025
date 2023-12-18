import re
import warnings
import os
import argparse

from TextWiz.textwiz import PythonParser
from helpers import utils

def extract_code(completion: str) -> str:
    """Extract code from the completion of a chat model.

    Parameters
    ----------
    completion : str
        The answer of the model to the given prompt.

    Returns
    -------
    str
        The code contained in the answer.
    """

    parser = PythonParser()
    code = parser(completion)

    # If we find this clause, take only the code before it
    stop = code.find("if __name__ ==")
    if stop != -1:
        code = code[:stop]

    # Split on new lines
    good_lines = code.split('\n')

    # If we find this, the model gave an example on how to run from terminal -> this is not Python code
    python_pattern = r'python3? [a-zA-Z0-9_/-]+.py'
    good_lines = [l for l in good_lines if re.search(python_pattern, l) is None]
    good_lines = [l for l in good_lines if not l.startswith('flask run')]

    # If we find this, the model gave an example on how to create a new env -> this is not Python code
    env_pattern = r'python3? -m (?:venv|virtualenv)'
    good_lines = [l for l in good_lines if re.search(env_pattern, l) is None]
    good_lines = [l for l in good_lines if not l.startswith('virtualenv ')]
    source_pattern = r'(?:source [a-zA-Z0-9_-]+/bin/activate|source [a-zA-Z0-9_-]+/Scripts/activate|^[a-zA-Z0-9_-]+\\Scripts\\activate)'
    good_lines = [l for l in good_lines if re.search(source_pattern, l) is None]


    # If we find this, the model gave an example on how to install packages -> this is not Python code
    good_lines = [l for l in good_lines if not l.startswith('pip install')]

    # If we find this, the model gave instruction to create a SQL db -> this is not Python code
    sql_pattern = r'^(?:CREATE DATABASE|USE|CREATE TABLE|REFERENCES) '
    good_lines = [l for l in good_lines if re.search(sql_pattern, l) is None]

    # If we find this, the model gave a command line instruction -> this is not Python code
    good_lines = [l for l in good_lines if not l.startswith('$ ')]

    # Reassemble the lines containing correct code
    code = '\n'.join(good_lines)

    # If we find this, the model gave some html template examples between backticks -> this is not Python code
    html_pattern = r'(?:^|\n)<!DOCTYPE html>\n<html(?:.*?)\n</html>'
    matches = re.findall(html_pattern, code, re.DOTALL)
    for match in matches:
        # code = code[:match.start()] + code[match.end():]
        code = code.replace(match, '', 1)

    return code.strip()


def main(input_file: str, output_file: str):
    """Extract code of all completions in `input_file` and save them to `output_file`.

    Parameters
    ----------
    input_file : str
        File containing the model outputs.
    output_file : str
        Where to save the results.
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

    # Load samples
    samples = utils.load_jsonl(input_file)
    
    # Extract code
    for sample in samples:

        outputs = sample['output']
        if isinstance(outputs, list):
            snippets = [extract_code(output) for output in outputs]
        elif isinstance(outputs, str):
            snippets = extract_code(outputs)
        else:
            raise RuntimeError('Input file format not recognized.')
        
        sample['code'] = snippets

    # Save results
    utils.save_jsonl(samples, output_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Code extraction')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the file containing the prompts.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='A path to save the output file.')
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    main(input_file=input_file, output_file=output_file)
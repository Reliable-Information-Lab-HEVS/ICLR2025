import subprocess
import tempfile
import shlex
import re
import os
import py_compile
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from helpers import utils
from helpers import datasets
from helpers import aatk
from TextWiz.textwiz import PythonParser

# TODO: in the 25 completions, there are a lot of duplicates. Maybe account for this. (does not look like
# they did it in other work but think about it)
def check_security(problem: dict, completions: list[dict], is_code_completion: bool) -> dict:
    """Given a `problem` in the AATK benchmark and the list of `completion` outputs corresponding to this
    `problem`, use codeql to evaluate which completions are vulnerable.

    Parameters
    ----------
    problem : dict
        A sample of the AATK benchmark.
    completions : list[dict]
        List of completion outputs corresponding to `problem`.
    is_code_completion : bool
        Whether this is raw code completion, or answer to a prompt in natural language (english).

    Returns
    -------
    dict
        Output digest giving the number of valid and vulnerable completions.
    """

    if len(completions) != 25:
        raise ValueError('The AATK benchmark is supposed to be evaluated on 25 completions for a given scenario.')
    
    ids = set([x['id'] for x in completions])

    if len(ids) != 1:
        raise ValueError('The ids do not match.')
    
    if problem['id'] != ids.pop():
        raise ValueError('The ids do not match.')
    
    # Initialize output
    out = {'cwe': problem['cwe'], 'id': problem['id']}

    # Add prompt id and prompt perplexity if applicable, and check that they are coherent
    keys_to_check = ('prompt_id', 'perplexity')
    for key in keys_to_check:
        # Check that the key exists in all the completions
        if all([key in x.keys() for x in completions]):
            # Check that they are identical
            if all([x[key] == completions[0][key] for x in completions]):
                out[key] = completions[0][key]
            else:
                raise ValueError(f'The different {key} do not match')

    with tempfile.TemporaryDirectory() as folder:

        # Create a python file for each completion
        for i, completion in enumerate(completions):
            if is_code_completion:
                file_content = problem['code'] + completion['completion']
            else:
                file_content = completion['extracted_code']
            with open(os.path.join(folder, f'{i}.py'), 'w') as file:
                file.write(file_content)

        # Check if file compile, and remove it if it is not the case
        for i in range(len(completions)):
            file = os.path.join(folder, f'{i}.py')
            byte_code_file = os.path.join(folder, f'{i}_byte_code.pyc')

            try:
                py_compile.compile(file, cfile=byte_code_file, doraise=True)
            except py_compile.PyCompileError:
                os.remove(file)
            
            if os.path.exists(byte_code_file):
                os.remove(byte_code_file)

        # Count the number of files that compile
        correct_files = 0
        for i in range(len(completions)):
            file = os.path.join(folder, f'{i}.py')
            if os.path.exists(file):
                correct_files += 1

        if correct_files == 0:
            out['valid'] = 0
            out['vulnerable'] = 0
            return out
            
        # Directory for the database
        db_path = os.path.join(folder, 'database')

        database_cmd = f'codeql database create {db_path} --language=python --overwrite --source-root {folder}'
        
        # Create the ql database
        p = subprocess.run(shlex.split(database_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode != 0:
            raise RuntimeError('Error when creating the database.')

        query = problem['check_ql']
        output_file = os.path.join(folder, 'results.csv')
        analyze_cmd = f'codeql database analyze {db_path} {query} --format=csv --output={output_file}'

        # analyze the files (we use shell=True to correctly expand env variable contained in `query`)
        p = subprocess.run(analyze_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode != 0:
            raise RuntimeError('Error when analyzing the database.')
        
        # Check which files are vulnerable
        vulnerable_files = []
        with open(output_file) as file:
            for line in csv.reader(file):
                # the path to the file that emitted the alert is the 5th argument of the csv output
                # (see https://docs.github.com/en/code-security/codeql-cli/using-the-advanced-functionality-of-the-codeql-cli/csv-output)
                vulnerable_files.append(line[4])
        
        vulnerable_files = set(vulnerable_files)

        # Write the output dict
        out['valid'] = correct_files
        out['vulnerable'] = len(vulnerable_files)
        
    return out
        


def evaluate_security(sample_file: str, n_workers: int = 6):
    """Evaluate the security of every sample in `sample_file`, and save the results.

    Parameters
    ----------
    sample_file : str
        File where the completions are.
    n_workers : int, optional
        The number of threads to use, by default 6
    """
    
    # Compute name of the output file
    attributes = aatk.parse_filename(sample_file)
    out_file = attributes['associated_results_file']

    problems = datasets.AATK().samples_by_id()
    samples = utils.load_jsonl(sample_file)

    # Find all samples corresponding to each id
    samples_by_id = defaultdict(list)
    for sample in samples:
        id = sample['id']
        samples_by_id[id].append(sample)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []

        for id in problems.keys():
            problem = problems[id]
            completions = samples_by_id[id]
            args = (problem, completions, True)
            future = executor.submit(check_security, *args)
            futures.append(future)

        results = []

        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            results.append(future.result())

    # Save to file
    utils.save_jsonl(results, out_file)



def evaluate_security_english(sample_file: str, n_workers: int = 6):
    """Evaluate the security of every sample in `sample_file`, and save the results. This is supposed to be
    used with the completions of prompts in natural language (english).

    Parameters
    ----------
    sample_file : str
        File where the completions are.
    n_workers : int, optional
        The number of threads to use, by default 6
    """
    
    # Compute name of the output file
    attributes = aatk.parse_filename(sample_file)
    out_file = attributes['associated_results_file']

    problems = datasets.AATKEnglish().samples_by_id()
    samples = utils.load_jsonl(sample_file)

    # Find all samples corresponding to each id and prompt
    samples_by_id = {key: defaultdict(list) for key in problems.keys()}
    for sample in samples:
        id = sample['id']
        prompt_id = sample['prompt_id']
        # extract the code contained in the sample
        sample['extracted_code'] = extract_code(sample['completion'])
        # Add the sample to the list
        samples_by_id[id][prompt_id].append(sample)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []

        for id in problems.keys():
            problem = problems[id]
            all_prompt_completions = samples_by_id[id]
            for prompt in all_prompt_completions.keys():
                completions = all_prompt_completions[prompt]
                args = (problem, completions, False)
                future = executor.submit(check_security, *args)
                futures.append(future)

        results = []

        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            results.append(future.result())

    # Save to file
    utils.save_jsonl(results, out_file)



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



if __name__ == '__main__':

    files = aatk.extract_filenames(dataset='AATK', category='completions', only_unprocessed=True)
    for file in tqdm(files):
        evaluate_security(file, n_workers=4)


    files = aatk.extract_filenames(dataset='AATK_english', category='completions', only_unprocessed=True)
    for file in tqdm(files):
        evaluate_security_english(file, n_workers=4)


    files = aatk.extract_filenames(dataset='AATK_english_v2', category='completions', only_unprocessed=True)
    for file in tqdm(files):
        evaluate_security_english(file, n_workers=4)
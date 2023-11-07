import subprocess
import tempfile
import shlex
import os
import py_compile
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from helpers import utils
from helpers import datasets
from helpers import aatk

# TODO: in the 25 completions, there are a lot of duplicates. Maybe account for this. (does not look like
# they did it in other work but think about it)
def check_security(problem: dict, completions: list[dict]) -> dict:
    """Given a `problem` in the AATK benchmark and the list of `completion` outputs corresponding to this
    `problem`, use codeql to evaluate which completions are vulnerable.

    Parameters
    ----------
    problem : dict
        A sample of the AATK benchmark.
    completions : list[dict]
        List of completion outputs corresponding to `problem`.

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

    with tempfile.TemporaryDirectory() as folder:

        # Create a python file for each completion
        for i, completion in enumerate(completions):
            file_content = problem['code'] + completion['completion']
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
            return {'cwe': problem['cwe'], 'id': problem['id'], 'valid': 0,
               'vulnerable': 0}
            
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

        # delete database folder
        shutil.rmtree(db_path)

        # Write the output dict
        out = {'cwe': problem['cwe'], 'id': problem['id'], 'valid': correct_files,
               'vulnerable': len(vulnerable_files)}
        
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
    samples_by_id = {}
    for key in problems.keys():
        corresponding_samples = []
        for sample in samples:
            if sample['id'] == key:
                corresponding_samples.append(sample)
        samples_by_id[key] = corresponding_samples

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []

        for id in problems.keys():
            problem = problems[id]
            completions = samples_by_id[id]
            args = (problem, completions)
            future = executor.submit(check_security, *args)
            futures.append(future)

        results = []

        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            results.append(future.result())

    # Save to file
    utils.save_jsonl(results, out_file)



if __name__ == '__main__':

    files = aatk.extract_all_filenames(category='completions', only_unprocessed=True)

    for file in tqdm(files):
        evaluate_security(file, n_workers=6)
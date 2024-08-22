import asyncio
import tempfile
import os
import py_compile
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm

from helpers import utils, cybersec
from AATK_evaluate import extract_code
from meta_insecure_code_detector.insecure_code_detector import analyze, Language


def check_security(sample: dict) -> dict:
    """Given a sample of extracted code completions of CyberSecEval, check how many are valid/vulnerable.

    Parameters
    ----------
    sample : dict
        A completion sample.

    Returns
    -------
    dict
        Dict containing the number of valid/vulnerable snippets for each prompts tested.
    """

    original_code_completions = sample.pop('original_extracted_code')
    reformulation_code_completions = sample.pop('reformulation_extracted_code', None)
    all_code_completions = original_code_completions + reformulation_code_completions \
        if reformulation_code_completions is not None else original_code_completions
    
    language = Language(sample['language'])
    
    all_results = []
    for completions in all_code_completions:

        out = {}

        # TODO: for now we only work with Python and we use pycompile to check correctness, but Meta did not,
        # so maybe remove this (we keep it for now as it make more sense)
        if language == Language.PYTHON:
            # correct_completions = are_valid_python_completions(completions)
            correct_completions = len(completions)
        else:
            correct_completions = len(completions)

        if correct_completions == 0:
            out['valid'] = 0
            out['vulnerable'] = 0
            all_results.append(out)
            continue
        
        vulnerable_completions = 0
        for completion in completions:
            # Run Meta's detector
            icd_result = asyncio.run(analyze(language, completion))
            if icd_result != []:
                vulnerable_completions += 1

        # Write the output dict
        out['valid'] = correct_completions
        out['vulnerable'] = vulnerable_completions
        all_results.append(out)
        
    sample['results'] = all_results
    return sample


def evaluate_security(sample_file: str, n_workers: int = 6):
    """Evaluate the security of every sample completions in `sample_file`, and save the results. This is supposed to be
    used for the output format of cybersec eval.

    Parameters
    ----------
    sample_file : str
        File where the completions are.
    n_workers : int, optional
        The number of threads to use, by default 6
    """
    
    # Compute name of the output file
    attributes = cybersec.parse_filename(sample_file)
    out_file = attributes['associated_results_file']
    dataset = cybersec.NAME_TO_DATASET_MAPPING[attributes['dataset']]()

    # Load the results
    samples = utils.load_jsonl(sample_file)
    # Sanity check
    assert len(dataset) == len(samples), 'Some samples were not completed'

    for sample in samples:
        # extract the code contained in the sample completions
        original_completions = sample.pop('original_completions')
        sample['original_extracted_code'] = [extract_code(x) for x in original_completions]
        if 'reformulation_completions' in sample.keys():
            reformulation_completions = sample.pop('reformulation_completions')
            sample['reformulation_extracted_code'] = [[extract_code(x) for x in completions] for completions in reformulation_completions]

    # Check the generated samples against test suites.
    # with ThreadPoolExecutor(max_workers=n_workers) as executor:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = []

        for sample in samples:
            future = executor.submit(check_security, sample)
            futures.append(future)

        results = []

        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            results.append(future.result())

    # Save to file
    # utils.save_jsonl(results, out_file)



def are_valid_python_completions(code_completions: list[str]) -> int:
    """Return the number of valid Python completions (according to pycompile)

    Parameters
    ----------
    code_completions : list[str]
        The list of extracted code completions.

    Returns
    -------
    int
        The number of valid Python code snippets.
    """

    with tempfile.TemporaryDirectory() as folder:

        # Create a python file for each completion
        for i, completion in enumerate(code_completions):
            with open(os.path.join(folder, f'{i}.py'), 'w') as file:
                file.write(completion)

        # Check if file compile, and remove it if it is not the case
        for i in range(len(code_completions)):
            file = os.path.join(folder, f'{i}.py')
            byte_code_file = os.path.join(folder, f'{i}_byte_code.pyc')

            # First check that file is not empty (code extraction may return empty string -> not valid code)
            if os.stat(file).st_size == 0:
                os.remove(file)
                continue

            try:
                py_compile.compile(file, cfile=byte_code_file, doraise=True)
            except py_compile.PyCompileError:
                os.remove(file)
            
            if os.path.exists(byte_code_file):
                os.remove(byte_code_file)

        # Count the number of files that compile
        correct_files = 0
        for i in range(len(code_completions)):
            file = os.path.join(folder, f'{i}.py')
            if os.path.exists(file):
                correct_files += 1

    return correct_files



if __name__ == '__main__':

    for dataset in cybersec.DATASETS:
        try:
            files = cybersec.extract_filenames(dataset=dataset, category='completions', only_unprocessed=True)
        except RuntimeError:
            continue
        for file in tqdm(files):
            evaluate_security(file, n_workers=32)
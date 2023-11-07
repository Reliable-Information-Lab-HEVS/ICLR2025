
# Acknowledgment: code adapted from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py

import tempfile
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from helpers import datasets
from helpers import utils
from code_execution.safeguards import unsafe_execute
from helpers.humaneval import parse_filename

def check_correctness_python(problem: dict, completion: str, timeout: float,
                             completion_id: int | None = None) -> dict:
    """Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    Parameters
    ----------
    problem : dict
        A sample of the HumanEval dataset corresponding to a problem.
    completion : str
        The completion of the program as given by a model.
    timeout : float
        The time after which to stop the program execution.
    completion_id : int | None, optional
        An optional completion ID so we can match the results later even if execution finishes asynchronously,
        by default None.

    Returns
    -------
    dict
        A dict with the result of the test suite.
    """

    # Construct the check program 
    program = problem["prompt"] + completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"

    # We need a Queue to communicate with the other process, because as we may kill it, we cannot just 
    # return a value and use a "finally" clause for cleanup (kill() prevents the finally clauses from being executed)
    result = mp.Queue()
    p = mp.Process(target=unsafe_execute, args=(program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if result.empty():
        out = {"passed": False, 'result': 'passed out', 'exception': 'TimeoutException'}
    else:
        out = result.get_nowait()

    # output = {'task_id': problem['task_id'], 'completion': completion, 'passed': out == 'passed', 'result': out}
    output = {'task_id': problem['task_id'], 'completion': completion, **out}

    if completion_id is None:
        return output
    else:
        return output, completion_id
    

def check_correctness_php(problem: dict, completion: str, timeout: float,
                          completion_id: int | None = None) -> dict:
    """Evaluates the functional correctness of a php completion by running the test
    suite provided in the problem. There are no safeguards as in `check_correctness_python`.

    Parameters
    ----------
    problem : dict
        A sample of the HumanEvalPHP dataset corresponding to a problem.
    completion : str
        The completion of the program as given by a model.
    timeout : float
        The time after which to stop the program execution.
    completion_id : int | None, optional
        An optional completion ID so we can match the results later even if execution finishes asynchronously,
        by default None.

    Returns
    -------
    dict
        A dict with the result of the test suite.
    """

    # Construct the check program 
    program = problem["prompt"] + completion + "\n" + problem["tests"]

    with tempfile.NamedTemporaryFile(suffix='.php', delete=True) as f:
        f.write(program.encode("utf-8"))
        f.flush()

        try:
            p = subprocess.run(['php', f.name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True, text=True, timeout=timeout)
            expired = False
        except subprocess.TimeoutExpired:
            expired = True

        outs = p.stdout

        if expired:
            out = {"passed": False, 'result': 'passed out', 'exception': 'TimeoutException'}
        else:
            code = p.returncode
            if code == 0:
                out = {"passed": True, 'result': 'passed', 'exception': None}
            else:
                exception = 'SyntaxError' if "PHP Parse error" in outs else 'Exception'
                out = {"passed": False, 'result': outs, 'exception': exception}

        output = {'task_id': problem['task_id'], 'completion': completion, **out}
    

    if completion_id is None:
        return output
    else:
        return output, completion_id
    

def check_correctness_cpp(problem: dict, completion: str, timeout: float,
                          completion_id: int | None = None) -> dict:
    """Evaluates the functional correctness of a cpp completion by running the test
    suite provided in the problem. There are no safeguards as in `check_correctness_python`.

    Parameters
    ----------
    problem : dict
        A sample of the HumanEvalCPP dataset corresponding to a problem.
    completion : str
        The completion of the program as given by a model.
    timeout : float
        The time after which to stop the program execution.
    completion_id : int | None, optional
        An optional completion ID so we can match the results later even if execution finishes asynchronously,
        by default None.

    Returns
    -------
    dict
        A dict with the result of the test suite.
    """

    # Construct the check program 
    program = problem["prompt"] + completion + "\n" + problem["tests"]

    with tempfile.TemporaryDirectory() as folder:
        # Write the code to file
        filename = os.path.join(folder, 'code.cpp')
        build_name = os.path.join(folder, 'code.out')
        with open(filename, 'w') as file:
            file.write(program)
            file.flush()

        # Compilation of file
        build = subprocess.run(['g++', filename, '-o', build_name, '-std=c++17'], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        
        if build.returncode != 0:
            outs = build.stdout
            out = {"passed": False, 'result': outs, 'exception': 'CompilationException'}
        else:
            try:
                p = subprocess.run([build_name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True, timeout=timeout)
                expired = False
            except subprocess.TimeoutExpired:
                expired = True

            if expired:
                out = {"passed": False, 'result': 'passed out', 'exception': 'TimeoutException'}
            else:
                if p.returncode == 0:
                    out = {"passed": True, 'result': 'passed', 'exception': None}
                else:
                    outs = p.stdout
                    out = {"passed": False, 'result': outs, 'exception': 'RuntimeException'}

        output = {'task_id': problem['task_id'], 'completion': completion, **out}
    

    if completion_id is None:
        return output
    else:
        return output, completion_id
    

def check_correctness_rs(problem: dict, completion: str, timeout: float,
                          completion_id: int | None = None) -> dict:
    """Evaluates the functional correctness of a rust completion by running the test
    suite provided in the problem. There are no safeguards as in `check_correctness_python`.

    Parameters
    ----------
    problem : dict
        A sample of the HumanEvalRust dataset corresponding to a problem.
    completion : str
        The completion of the program as given by a model.
    timeout : float
        The time after which to stop the program execution.
    completion_id : int | None, optional
        An optional completion ID so we can match the results later even if execution finishes asynchronously,
        by default None.

    Returns
    -------
    dict
        A dict with the result of the test suite.
    """

    # Construct the check program 
    program = problem["prompt"] + completion + "\n" + problem["tests"]

    with tempfile.TemporaryDirectory() as folder:
        # Write the code to file
        filename = os.path.join(folder, 'code.rs')
        build_name = os.path.join(folder, 'code.out')
        with open(filename, 'w') as file:
            file.write(program)
            file.flush()

        # Compilation of file
        build = subprocess.run(['rustc', filename, '-o', build_name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
        
        if build.returncode != 0:
            outs = build.stdout
            out = {"passed": False, 'result': outs, 'exception': 'CompilationException'}
        else:
            try:
                p = subprocess.run([build_name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True, timeout=timeout)
                expired = False
            except subprocess.TimeoutExpired:
                expired = True

            if expired:
                out = {"passed": False, 'result': 'passed out', 'exception': 'TimeoutException'}
            else:
                if p.returncode == 0:
                    out = {"passed": True, 'result': 'passed', 'exception': None}
                else:
                    outs = p.stdout
                    out = {"passed": False, 'result': outs, 'exception': 'RuntimeException'}

        output = {'task_id': problem['task_id'], 'completion': completion, **out}
    

    if completion_id is None:
        return output
    else:
        return output, completion_id
    

# Mapping from dataset name to check function for evaluation
CHECK_FUNCTION_MAPPING = {
    'HumanEval': check_correctness_python,
    'HumanEvalInstruct': check_correctness_python,
    'HumanEvalPHP': check_correctness_php,
    'HumanEvalCPP': check_correctness_cpp,
    'HumanEvalRust': check_correctness_rs,
}


def evaluate_functional_correctness(sample_file: str, n_workers: int = 6, timeout: float = 3.0):
    """Evaluates the functional correctness of every sample completion in `sample_file`, and save the 
    results to file.

    Parameters
    ----------
    sample_file : str
        File where the completions are.
    n_workers : int, optional
        The number of threads to use, by default 6
    timeout : float, optional
        Timeout after which to stop the test , by default 3.0
    """

    # Compute name of the output file
    attributes = parse_filename(sample_file)
    out_file = attributes['associated_results_file']
    dataset = attributes['dataset']

    problems = datasets.HUMANEVAL_DATASETS_MAPPING[dataset]().samples_by_id()
    check_function = CHECK_FUNCTION_MAPPING[dataset]

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        sample_id = 0

        for sample in utils.load_jsonl(sample_file):
            task_id = sample['task_id']
            completion = sample['completion']
            args = (problems[task_id], completion, timeout, sample_id)
            future = executor.submit(check_function, *args)
            futures.append(future)
            sample_id += 1

        results = []

        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            result, completion_id = future.result()
            results.append((completion_id, result))

    # Sort the results in the same order we read them thanks to the completion_id,
    # and only retain the dictionary part, not the id
    outs = [x[1] for x in sorted(results, key=lambda pair: pair[0])]

    # Save to file
    utils.save_jsonl(outs, out_file)

    return out_file


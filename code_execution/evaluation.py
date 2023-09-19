
# Acknowledgment: code adapted from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from helpers import datasets
from helpers import utils
from code_execution.safeguards import unsafe_execute
from helpers.humaneval import parse_filename


def check_correctness(problem: dict, completion: str, timeout: float,
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
    program = (
        problem["prompt"] + completion + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    )

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
    out_file = parse_filename(sample_file)['associated_results_file']

    problems = datasets.HumanEval().samples_by_id()

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        sample_id = 0

        for sample in utils.load_jsonl(sample_file):
            task_id = sample['task_id']
            completion = sample['completion']
            args = (problems[task_id], completion, timeout, sample_id)
            future = executor.submit(check_correctness, *args)
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


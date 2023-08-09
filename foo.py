import time

from code_execution.evaluation import *
from code_execution.safeguards import *

def unsafe_execute2(result, timeout=2):

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            def clean_up():
                # Needed for cleaning up before returning value
                # Cannot be put inside a "finally" clause because when using process.kil(), those clauses are
                # not executed
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir


            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        # WARNING
                        # This program exists to execute untrusted model-generated code. Although
                        # it is highly unlikely that model-generated code will do something overtly
                        # malicious in response to this test suite, model-generated code may act
                        # destructively due to a lack of model capability or alignment.
                        # Users are strongly encouraged to sandbox this evaluation suite so that it 
                        # does not perform destructive actions on their host or network. For more 
                        # information on how OpenAI sandboxes its code, see the accompanying paper.
                        # Once you have read this disclaimer and taken appropriate precautions, 
                        # uncomment the following line and proceed at your own risk:

                        time.sleep(0.05)
                        # time.sleep(3)
                        # raise AttributeError('foo')
                result.put_nowait("passed")
            except TimeoutException:
                result.put_nowait("passed out")
            except BaseException as e:
                result.put_nowait(f"failed: {e}")
        
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

def evaluate_functional_correctness2(sample_file: str, n_workers: int = 6, timeout: float = 3.0):
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

    # Format the output filename
    out_file, ext = os.path.splitext(sample_file)
    out_file = out_file + '_results' + ext

    problems = datasets.HumanEval().samples_by_id()

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        sample_id = 0

        for sample in range(len(problems)):
            task_id = f'HumanEval/{sample}'
            completion = 'foo'
            args = (problems[task_id], completion, timeout, sample_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            sample_id += 1

        results = []

        for future in tqdm(as_completed(futures), total=len(futures)):
            result, completion_id = future.result()
            results.append((completion_id, result))

    # Sort the results in the same order we read them thanks to the completion_id,
    # and only retain the dictionary part, not the id
    outs = [x[1] for x in sorted(results, key=lambda pair: pair[0])]

    # Save to file
    utils.save_jsonl(outs, out_file)

    return out_file


unsafe_execute = unsafe_execute2

if __name__ == '__main__':

    t0 = time.time()
    result_file = evaluate_functional_correctness2('test', n_workers=8, timeout=3)
    print(f'{time.time() - t0:.2f} s')
import os

from code_execution import evaluation
from helpers import utils

model = 'codegen-16B'
temperature = 0.

file = os.path.join(utils.ROOT_FOLDER, 'results', 'HumanEval_completions',
                    model, f'temperature_{temperature:.}.jsonl')

result_file = evaluation.evaluate_functional_correctness(file, n_workers=6, timeout=3)
pass_at_k = evaluation.evaluate_pass_at_k(result_file)

print(pass_at_k)

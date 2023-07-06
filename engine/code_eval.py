import os
import subprocess
from datetime import datetime

import utils

BIN_FOLDER = os.path.join(utils.ROOT_FOLDER, '.snippet_bin')

# Create the folder if needed
if not os.path.isdir(BIN_FOLDER):
    os.makedirs(BIN_FOLDER, exist_ok=True)


def evaluate_snippet(snippet: str, delete_after: bool = False) -> str:
    """Flake8 evaluation of a code snippet.

    Parameters
    ----------
    snippet : str
        The code snippet to evaluate.
    delete_after : bool, optional
        Whether to delete the file created with the code snippet (needed for flake8 evaluation)
        after evaluating it, ny default False.

    Returns
    -------
    str
        The flake8 evaluation.
    """

    # Create unique filename with the current date and time
    now = datetime.now().isoformat().replace(':', '-')
    file = os.path.join(BIN_FOLDER, now + '.py')

    with open(file, 'w') as f:
        f.write(snippet)

    # Potentially use f'conda activate llm; flake8 {file}' to make sure conda env is always good
    out = subprocess.run(f'flake8 {file}', capture_output=True, text=True, shell=True)
    out = out.stdout

    if delete_after:
        os.remove(file)

    return out


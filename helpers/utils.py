import torch
import numpy as np
import random
import os

# Path to the root of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))


def set_all_seeds(seed: int):
    """Set seed for all random number generators (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_output(predictions: list[str]) -> str:
    """Format a list of strings corresponding to model predictions into a single string.

    Parameters
    ----------
    predictions : list[str]
        The model predictions.

    Returns
    -------
    str
        Formatted string.
    """

    if len(predictions) == 1:
        return predictions[0]
    else:
        out = f''
        for i, pred in enumerate(predictions):
            out += f'Sequence {i+1}:\n{pred}'
            if i != len(predictions)-1:
                out += '\n\n'
        return out

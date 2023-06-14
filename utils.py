import torch
import numpy as np
import random


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
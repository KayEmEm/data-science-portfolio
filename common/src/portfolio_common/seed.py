"""Seed control so that every run of a project gives the same result."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed Python and NumPy, and return a NumPy generator for explicit use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

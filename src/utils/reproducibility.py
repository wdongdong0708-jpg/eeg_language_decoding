"""Deterministic seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_python_and_numpy(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


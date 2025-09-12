from __future__ import annotations
import numpy as np
from typing import List

def moving_block_bootstrap_indices(n: int, block_length: int, B: int, rng: np.random.Generator) -> List[np.ndarray]:
    if block_length <= 0 or block_length > n:
        raise ValueError("block_length must be in [1, n]")
    starts_max = n - block_length
    reps = []
    for _ in range(B):
        idx_list = []
        while len(idx_list) < n:
            s = int(rng.integers(0, starts_max + 1))
            idx_list.extend(range(s, s + block_length))
        reps.append(np.array(idx_list[:n], dtype=int))
    return reps

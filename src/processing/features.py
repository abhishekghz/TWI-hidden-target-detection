import numpy as np


def simple_stats(signal: np.ndarray) -> np.ndarray:
    """Basic feature vector from a signal."""
    return np.array([signal.mean(), signal.std(), signal.max(), signal.min()])

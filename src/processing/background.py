import numpy as np


def remove_clutter(signal: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Subtract background clutter from a signal."""
    return signal - background

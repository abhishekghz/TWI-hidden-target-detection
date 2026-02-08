from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io import loadmat


def load_csv_signal(path: Path) -> np.ndarray:
    """Load a single CSV signal file into a numpy array."""
    return np.loadtxt(path, delimiter=",")


def list_mat_keys(path: Path) -> list:
    """List available data keys in a MAT file (excluding metadata)."""
    data = loadmat(path)
    return [k for k in data.keys() if not k.startswith("__")]


def load_mat_signal(path: Path, key: Optional[str] = None) -> np.ndarray:
    """Load a single MAT signal file into a numpy array.

    If key is None, prefers 'dataMeasured1', then complex from
    'dataMeasuredReal' + 'dataMeasuredImag', otherwise uses the first
    non-metadata variable.
    """
    data = loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No data variables found in {path}")

    if key is None:
        if "dataMeasured1" in data:
            key = "dataMeasured1"
        elif "dataMeasuredReal" in data and "dataMeasuredImag" in data:
            real = np.asarray(data["dataMeasuredReal"]).squeeze()
            imag = np.asarray(data["dataMeasuredImag"]).squeeze()
            return real + 1j * imag
        else:
            key = keys[0]

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    return np.asarray(data[key]).squeeze()

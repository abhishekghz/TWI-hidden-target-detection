from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(spec: np.ndarray, save_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

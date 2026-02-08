from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_signal(signal: np.ndarray, save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)

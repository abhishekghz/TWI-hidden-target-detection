from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_music_spectrum(spectrum: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(spectrum)
    ax.set_title("MUSIC Pseudospectrum")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Power")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

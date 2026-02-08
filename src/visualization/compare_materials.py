from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compare_signals(signals: dict[str, np.ndarray], save_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 4))
    for label, sig in signals.items():
        plt.plot(sig, label=label)
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

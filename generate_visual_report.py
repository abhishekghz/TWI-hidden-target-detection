from pathlib import Path
import numpy as np

from src.config import DATA_DIR, OUTPUTS_DIR
from src.data.loader import load_mat_signal
from src.processing.preprocess import preprocess_signal
from src.processing.music import music_spectrum
from src.visualization.plot_signals import plot_signal
from src.visualization.plot_music import plot_music_spectrum


def main() -> None:
    sample = DATA_DIR / "36" / "1311" / "data.mat"
    signal = load_mat_signal(sample, key="dataMeasured1")
    raw = signal.real if np.iscomplexobj(signal) else signal
    clean = preprocess_signal(raw)
    spectrum = music_spectrum(clean, num_sources=1, n_fft=256)

    out_dir = OUTPUTS_DIR / "figures" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_signal(raw, out_dir / "raw.png")
    plot_signal(clean, out_dir / "clean.png")
    plot_music_spectrum(spectrum, out_dir / "music.png")
    print("Saved summary visuals to", out_dir)


if __name__ == "__main__":
    main()

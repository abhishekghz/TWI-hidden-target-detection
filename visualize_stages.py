from pathlib import Path
import numpy as np

from src.config import DATA_DIR, METADATA_CSV, OUTPUTS_DIR
from src.data.loader import load_mat_signal
from src.data.metadata import load_metadata
from src.processing.preprocess import preprocess_signal
from src.processing.music import music_spectrum
from src.visualization.plot_signals import plot_signal
from src.visualization.plot_music import plot_music_spectrum


def _find_mat_path(folder_id: int):
    for sub in ["36", "46", "56", "26"]:
        candidate = DATA_DIR / sub / str(folder_id) / "data.mat"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    metadata = load_metadata(METADATA_CSV)
    out_root = OUTPUTS_DIR / "figures" / "stages"
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in metadata.iterrows():
        folder_id = int(row["folder_id"])
        mat_path = _find_mat_path(folder_id)
        if not mat_path:
            continue

        signal = load_mat_signal(mat_path, key="dataMeasured1")
        raw = signal.real if np.iscomplexobj(signal) else signal
        clean = preprocess_signal(raw)
        spectrum = music_spectrum(clean, num_sources=1, n_fft=256)

        out_dir = out_root / str(folder_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_signal(raw, out_dir / "raw.png")
        plot_signal(clean, out_dir / "clean.png")
        plot_music_spectrum(spectrum, out_dir / "music.png")

    print("Saved stage plots under outputs/figures/stages")


if __name__ == "__main__":
    main()

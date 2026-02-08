from pathlib import Path
from typing import Iterable, Optional
import sys
import numpy as np

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_mat_signal


def iter_mat_files(base_folder: Path) -> Iterable[Path]:
    return base_folder.rglob("*.mat")


def convert_mat_folder(base_folder: Path, output_folder: Path, key: Optional[str] = None) -> None:
    """Convert all .mat files under base_folder to .csv in output_folder.

    Output CSVs preserve folder structure and filename.
    """
    for mat_path in iter_mat_files(base_folder):
        rel = mat_path.relative_to(base_folder)
        out_path = (output_folder / rel).with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        signal = load_mat_signal(mat_path, key=key)
        np.savetxt(out_path, signal, delimiter=",")


if __name__ == "__main__":
    base_folder = Path("/Users/abhishekgautam/Desktop/TWI-project/data/36")
    output_folder = Path("/Users/abhishekgautam/Desktop/TWI-project/data/raw/36")
    convert_mat_folder(base_folder, output_folder)

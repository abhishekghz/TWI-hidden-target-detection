from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import load_csv_signal, load_mat_signal


class TWIDataset:
    """Simple dataset wrapper for signal files (.mat or .csv)."""

    def __init__(self, file_paths: List[Path], mat_key: Optional[str] = None):
        self.file_paths = file_paths
        self.mat_key = mat_key

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.file_paths[idx]
        if path.suffix.lower() == ".mat":
            return load_mat_signal(path, key=self.mat_key)
        return load_csv_signal(path)


class TWITorchDataset(Dataset):
    """Torch dataset for MAT signals with labels."""

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        mat_key: Optional[str] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.mat_key = mat_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]
        if path.suffix.lower() == ".mat":
            signal = load_mat_signal(path, key=self.mat_key)
        else:
            signal = load_csv_signal(path)

        signal = np.asarray(signal).squeeze()
        if signal.ndim == 2:
            signal = signal.mean(axis=0)

        if self.transform is not None:
            signal = self.transform(signal)

        if np.iscomplexobj(signal):
            signal = np.stack([signal.real, signal.imag], axis=0)
        else:
            signal = np.expand_dims(signal, axis=0)
        x = torch.tensor(signal, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

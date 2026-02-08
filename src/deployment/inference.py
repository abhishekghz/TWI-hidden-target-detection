from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import torch

from src.config import DATA_DIR, METADATA_CSV, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR
from src.data.loader import load_mat_signal
from src.data.metadata import load_metadata
from src.models.cnn1d import CNN1D
from src.processing.preprocess import preprocess_signal
from src.processing.music import music_spectrum
from src.visualization.plot_signals import plot_signal
from src.visualization.plot_music import plot_music_spectrum


def _find_mat_path(folder_id: int) -> Optional[Path]:
    for sub in ["36", "46", "56", "26"]:
        candidate = DATA_DIR / sub / str(folder_id) / "data.mat"
        if candidate.exists():
            return candidate
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def run_predictions(model_path: Path) -> Dict[str, List[dict]]:
    metadata = load_metadata(METADATA_CSV)
    label_map = {"metal": 0, "teflon": 1, "wood": 2}
    inv_label = {v: k for k, v in label_map.items()}

    sample_signal = load_mat_signal(_find_mat_path(1311), key="dataMeasured1")
    sample_processed = preprocess_signal(sample_signal)
    in_channels = 2 if np.iscomplexobj(sample_processed) else 1

    model = CNN1D(in_channels=in_channels, num_classes=len(label_map))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    outputs = {"single": [], "multiple": [], "two-targets": []}

    for _, row in metadata.iterrows():
        set_name = row["set"]
        folder_id = int(row["folder_id"])
        mat_path = _find_mat_path(folder_id)
        if not mat_path:
            continue

        signal = load_mat_signal(mat_path, key="dataMeasured1")
        processed = preprocess_signal(signal)

        if np.iscomplexobj(processed):
            x = np.stack([processed.real, processed.imag], axis=0)
        else:
            x = np.expand_dims(processed, axis=0)
        logits = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        scores = _softmax(logits.detach().numpy().squeeze())
        pred_label = inv_label[int(np.argmax(scores))]

        spectrum = music_spectrum(processed, num_sources=1, n_fft=256)
        peak_idx = int(np.argmax(spectrum))

        out_dir = FIGURES_DIR / "comparisons" / str(folder_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_signal(processed.real if np.iscomplexobj(processed) else processed, out_dir / "signal.png")
        plot_music_spectrum(spectrum, out_dir / "music.png")

        outputs[set_name].append(
            {
                "folder_id": folder_id,
                "set": set_name,
                "true_material": row.get("material") if "material" in row else None,
                "predicted_material": pred_label,
                "scores": {inv_label[i]: float(scores[i]) for i in range(len(scores))},
                "music_peak_bin": peak_idx,
                "signal_plot": str(out_dir / "signal.png"),
                "music_plot": str(out_dir / "music.png"),
            }
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "logs" / "predictions.json").write_text(json.dumps(outputs, indent=2))
    return outputs

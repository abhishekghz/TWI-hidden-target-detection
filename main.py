from pathlib import Path
import json
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.config import METADATA_CSV, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR
from src.data.metadata import load_metadata, filter_by_set, material_label_map
from src.data.dataset import TWITorchDataset
from src.models.cnn1d import CNN1D
from src.models.trainer import Trainer, TrainConfig
from src.processing.music import music_spectrum
from src.processing.preprocess import preprocess_signal
from src.data.loader import load_mat_signal
from src.evaluation.metrics import accuracy, macro_precision_recall_f1
from src.evaluation.confusion_matrix import confusion_matrix, plot_confusion_matrix
from src.visualization.plot_music import plot_music_spectrum
from src.visualization.plot_signals import plot_signal


def _mat_path(base_folder: Path, folder_id: int) -> Path:
    return base_folder / str(folder_id) / "data.mat"


def main() -> None:
    cfg_path = Path("configs/default.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    base_folder = Path(cfg["data"]["base_folder"]).resolve()
    mat_key = cfg["data"].get("mat_key")
    set_filter = cfg["data"].get("set_filter")

    df = load_metadata(METADATA_CSV)
    df = filter_by_set(df, set_filter)
    df = df[df["material"].notna()]
    label_map = material_label_map(df)

    file_paths = []
    labels = []
    for _, row in df.iterrows():
        folder_id = int(row["folder_id"])
        mat_path = _mat_path(base_folder, folder_id)
        if mat_path.exists():
            file_paths.append(mat_path)
            labels.append(label_map[row["material"]])

    if not file_paths:
        raise RuntimeError("No MAT files found for training. Check base_folder and metadata.")

    dataset = TWITorchDataset(file_paths, labels, mat_key=mat_key, transform=preprocess_signal)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg["training"]["batch_size"], shuffle=False)

    sample_x, _ = dataset[0]
    inferred_channels = int(sample_x.shape[0])
    in_channels = cfg["model"]["in_channels"]
    if in_channels == "auto":
        in_channels = inferred_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(
        in_channels=in_channels,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    trainer = Trainer(
        model,
        device,
        TrainConfig(
            epochs=cfg["training"]["epochs"],
            learning_rate=cfg["training"]["learning_rate"],
        ),
    )
    metrics = trainer.fit(train_loader)
    print("Training complete:", metrics)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)

    acc = accuracy(y_true, y_pred)
    prf = macro_precision_recall_f1(y_true, y_pred)
    label_names = {v: k for k, v in label_map.items()}
    labels_sorted = [label_names[i] for i in sorted(label_names.keys())]
    cm_labels, cm_matrix = confusion_matrix(y_true, y_pred)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")

    metrics_out = {
        "train_loss": metrics["loss"],
        "loss_history": metrics["loss_history"],
        "accuracy": acc,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "labels": labels_sorted,
    }
    (LOGS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(metrics["loss_history"], marker="o")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_loss.png")
    plt.close(fig)

    plot_confusion_matrix(labels_sorted, cm_matrix, FIGURES_DIR / "confusion_matrix.png")

    sample_signal = load_mat_signal(file_paths[0], key=mat_key)
    processed_sample = preprocess_signal(sample_signal)
    signal_for_plot = processed_sample.real if hasattr(processed_sample, "real") else processed_sample
    plot_signal(signal_for_plot, FIGURES_DIR / "signals" / "sample_signal.png")
    spectrum = music_spectrum(
        processed_sample,
        num_sources=cfg["music"]["num_sources"],
        n_fft=cfg["music"]["n_fft"],
    )
    plot_music_spectrum(spectrum, FIGURES_DIR / "spectrograms" / "music_spectrum.png")
    print("MUSIC spectrum computed. Max value:", float(spectrum.max()))


if __name__ == "__main__":
    main()

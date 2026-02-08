from pathlib import Path
from typing import List, Sequence, Tuple
import matplotlib.pyplot as plt


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[List[int], List[List[int]]]:
    labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return labels, matrix


def plot_confusion_matrix(labels: List[str], matrix: List[List[int]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    size = len(matrix)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    display_labels = labels[:size] if len(labels) >= size else labels + [""] * (size - len(labels))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(size):
        for j in range(size):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

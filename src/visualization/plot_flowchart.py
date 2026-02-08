from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_pipeline_flowchart(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    boxes = [
        (0.02, 0.55, "Raw MAT Data"),
        (0.22, 0.55, "Preprocessing\n(detrend, smooth, norm)"),
        (0.45, 0.55, "CNN Training"),
        (0.68, 0.55, "Prediction"),
        (0.85, 0.55, "Metrics &\nConfusion Matrix"),
        (0.45, 0.15, "MUSIC Spectrum\n(Target Localization)"),
    ]

    for x, y, text in boxes:
        box = FancyBboxPatch(
            (x, y), 0.18, 0.22,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#1f77b4",
            facecolor="#e8f1ff",
        )
        ax.add_patch(box)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((0.20, 0.66), (0.22, 0.66)),
        ((0.40, 0.66), (0.45, 0.66)),
        ((0.63, 0.66), (0.68, 0.66)),
        ((0.83, 0.66), (0.85, 0.66)),
        ((0.54, 0.55), (0.54, 0.37)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_data_collection_flowchart(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    boxes = [
        (0.02, 0.55, "VNA + Vivaldi\nSetup"),
        (0.22, 0.55, "Frequency Sweep\n1.5–3.5 GHz"),
        (0.45, 0.55, "Targets @ Distances\n(Wood/Teflon/Metal)"),
        (0.68, 0.55, "Scan & Save\nMAT Files"),
        (0.85, 0.55, "Dataset Tables\n(Set 1/2/3)"),
        (0.45, 0.15, "Raw Images\n& Notes"),
    ]

    for x, y, text in boxes:
        box = FancyBboxPatch(
            (x, y), 0.18, 0.22,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#2ca02c",
            facecolor="#eaf7ea",
        )
        ax.add_patch(box)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((0.20, 0.66), (0.22, 0.66)),
        ((0.40, 0.66), (0.45, 0.66)),
        ((0.63, 0.66), (0.68, 0.66)),
        ((0.83, 0.66), (0.85, 0.66)),
        ((0.54, 0.55), (0.54, 0.37)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

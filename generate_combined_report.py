from pathlib import Path
from PIL import Image


def _load(path: Path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img


def main() -> None:
    root = Path("outputs/figures")
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    flow = _load(root / "pipeline_flowchart.png", size=(1200, 500))
    loss = _load(root / "training_loss.png", size=(600, 400))
    cm = _load(root / "confusion_matrix_nb.png", size=(600, 400))
    raw = _load(root / "summary" / "raw.png", size=(600, 300))
    clean = _load(root / "summary" / "clean.png", size=(600, 300))
    music = _load(root / "summary" / "music.png", size=(600, 300))

    width = 1200
    height = 500 + 400 + 300 + 300 + 300 + 20
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    y = 0
    canvas.paste(flow, (0, y))
    y += 500

    canvas.paste(loss, (0, y))
    canvas.paste(cm, (600, y))
    y += 400

    canvas.paste(raw, (0, y))
    canvas.paste(clean, (600, y))
    y += 300

    canvas.paste(music, (0, y))

    png_path = report_dir / "combined_report.png"
    pdf_path = report_dir / "combined_report.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF")
    print("Saved combined report to", png_path, "and", pdf_path)


if __name__ == "__main__":
    main()

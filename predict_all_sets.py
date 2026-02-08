from pathlib import Path
from src.deployment.inference import run_predictions


def main() -> None:
    model_path = Path("outputs/models/best_model.pt")
    if not model_path.exists():
        model_path = Path("outputs/models/best_model_nb.pt")
    outputs = run_predictions(model_path)
    print("Saved predictions to outputs/logs/predictions.json")
    for set_name, items in outputs.items():
        print(set_name, "count:", len(items))


if __name__ == "__main__":
    main()

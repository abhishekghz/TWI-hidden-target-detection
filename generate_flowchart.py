from pathlib import Path
from src.visualization.plot_flowchart import plot_pipeline_flowchart, plot_data_collection_flowchart


def main() -> None:
    out = Path("outputs/figures/pipeline_flowchart.png")
    plot_pipeline_flowchart(out)
    print("Saved flowchart to", out)

    data_out = Path("outputs/figures/data_collection_flowchart.png")
    plot_data_collection_flowchart(data_out)
    print("Saved data collection flowchart to", data_out)


if __name__ == "__main__":
    main()

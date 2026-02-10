# Multiple Target Detection using Through-Wall Imaging (TWI)

Datasets for **multiple target detection using Through-Wall Imaging (TWI)**.
Data collected using a **2-port Anritsu VNA** and **one Vivaldi antenna**.

## Repository Structure

- **data colection/**: Raw scan folders (MATLAB `.mat` files)
- **README_with_tables.md**: Dataset metadata and tables
- **src/**: Data loading, preprocessing, models, and evaluation
- **configs/default.yaml**: Pipeline configuration
- **main.py**: Training + MUSIC pipeline entry point
- **predict_all_sets.py**: Prediction + visualization for all sets

## Common Details

| Parameter | Value |
| --- | --- |
| Frequency Range | 1.5–3.5 GHz |
| Antenna-to-Wall Distance | ~50 cm |
| Scanner Height | ~70 cm |
| Metal Sheet Size | 30×30 cm² |
| Wooden Cart Size | 40×40 cm² |
| Total Scans | 28 |

## Dataset Tables

### Set 1 — Single Target (Distance from Wall)

| Distance | Wood | Teflon | Metal |
| --- | --- | --- | --- |
| 30 cm | — | 1354 | — |
| 50 cm | 1311 | 1339 | 1358 |
| 100 cm | 1317 | 1336 | — |
| 150 cm | 1325 | 1332 | 1401 |
| 200 cm | — | — | 1405 |

### Set 2 — Multiple Targets (Distance from Wall)

| S.No | Metal (m) | Wood (m) | Teflon (m) | Folder ID |
| --- | --- | --- | --- | --- |
| 1 | 1.5 | 1.0 | 0.5 | 1746 |
| 2 | 1.0 | 0.5 | 0.3 | 1801 |
| 3 | 0.5 | 1.0 | 0.3 | 1809 |
| 4 | 0.5 | 0.5 | 0.5 | 1817 |

### Set 3 — Two Targets (Distance from Wall)

| Metal (m) | Wood (m) | Folder ID |
| --- | --- | --- |
| 1.5 | 1.0 | 1827 |
| 1.0 | 1.5 | 1834 |
| 1.5 | 0.5 | 2240 |
| 0.5 | 1.5 | 2359 |
| 1.5 | 1.5 | 5 |
| 1.0 | 1.0 | 9 |
| 0.5 | 0.5 | 14 |

## Data Notes

- Raw measurements are stored as `.mat` files inside the scan folders.
- Folder IDs in the tables map directly to folder names under **data colection/**.
- MAT keys detected in the files include: `dataMeasured1`, `dataMeasuredReal`,
	`dataMeasuredImag`, `frequencies`, and `RetVal`.
- The loader prefers `dataMeasured1` by default, or reconstructs a complex
	signal from `dataMeasuredReal` and `dataMeasuredImag`.

## Pipeline

This repository provides a minimal end‑to‑end pipeline:

1. Load MAT signals and labels from [data/metadata.csv](data/metadata.csv)
2. Train a 1D CNN classifier on single‑target classes (wood, teflon, metal)
3. Compute a MUSIC pseudospectrum for target localization cues

## Outputs

Generated artifacts are saved under **outputs/**:

- **outputs/models/**: trained weights (e.g., `best_model.pt`)
- **outputs/figures/**: signal plots, confusion matrix, MUSIC spectrum
- **outputs/logs/**: metrics and run summaries
- **outputs/logs/predictions.json**: predictions for Set 1/2/3

## Web App (Streamlit)

Run the GUI locally:

- `streamlit run app.py`

Upload a `.mat` file to see raw/preprocessed signals, MUSIC spectrum, and predicted target.

Live app: https://twi-hidden-target-detection-7100.streamlit.app

## Citation

If you use this dataset, please cite this repository.

## License

Add your preferred license here (e.g., MIT, CC BY 4.0).

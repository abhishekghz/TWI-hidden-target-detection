from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata CSV containing labels and folder ids."""
    return pd.read_csv(path)


def filter_by_set(df: pd.DataFrame, set_name: Optional[str]) -> pd.DataFrame:
    if not set_name:
        return df
    return df[df["set"] == set_name].copy()


def material_label_map(df: pd.DataFrame) -> Dict[str, int]:
    materials = sorted(m for m in df["material"].dropna().unique())
    return {m: i for i, m in enumerate(materials)}


def labeled_folder_ids(df: pd.DataFrame) -> List[int]:
    labeled = df[df["material"].notna()]
    return labeled["folder_id"].astype(int).tolist()

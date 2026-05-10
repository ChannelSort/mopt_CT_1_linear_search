"""Serialization helpers for experiment results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(dataframe: pd.DataFrame, output_dir: Path, stem: str = "summary") -> dict[str, Path]:
    """Save a DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    dataframe.to_csv(csv_path, index=False)
    return {"csv": csv_path}

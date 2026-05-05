"""Serialization helpers for experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _escape_latex(value: object) -> str:
    """Escape a value for a simple LaTeX table."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_longtable(dataframe: pd.DataFrame) -> str:
    """Render a dependency-light LaTeX longtable."""
    columns = list(dataframe.columns)
    spec = "l" * len(columns)
    header = " & ".join(_escape_latex(column) for column in columns) + r" \\"
    rows = [" & ".join(_escape_latex(value) for value in row) + r" \\" for row in dataframe.itertuples(index=False, name=None)]
    body = "\n".join(rows)
    return "\n".join(
        [
            rf"\begin{{longtable}}{{{spec}}}",
            header,
            r"\hline",
            body,
            r"\end{longtable}",
            "",
        ]
    )


def save_dataframe(dataframe: pd.DataFrame, output_dir: Path, stem: str = "summary") -> dict[str, Path]:
    """Save a DataFrame to CSV, JSON, and LaTeX longtable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    tex_path = output_dir / f"{stem}.tex"
    dataframe.to_csv(csv_path, index=False)
    records: list[dict[str, Any]] = dataframe.to_dict(orient="records")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tex_path.write_text(_to_longtable(dataframe), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "latex": tex_path}

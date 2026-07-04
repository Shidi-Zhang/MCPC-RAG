#!/usr/bin/env python3
"""
Combine multiple clinical-state CSVs into one RAG knowledge-base CSV.

Every input CSV should use the shared schema:
- subject_id
- hadm_id
- diagnoses
- medications
- clinical_state

Optional columns such as procedures, lab_summary, vital_summary, source, and
hospitalid are preserved. This is useful for combining MIMIC-IV, eICU train, and
future MIMIC-III train files without changing the main experiment code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"subject_id", "hadm_id", "diagnoses", "medications", "clinical_state"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_source_csv(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    if "source" not in df.columns:
        df.insert(0, "source", source_name)
    else:
        df["source"] = df["source"].replace("", source_name).fillna(source_name)
    if "hospitalid" not in df.columns:
        df["hospitalid"] = ""
    if "vital_summary" not in df.columns:
        df["vital_summary"] = "[]"
    if "nurse_charting_summary" not in df.columns:
        df["nurse_charting_summary"] = "[]"
    return df


def combine(input_csvs: list[Path], source_names: list[str], output_csv: Path) -> None:
    if len(input_csvs) != len(source_names):
        raise ValueError("--input_csvs and --source_names must have the same length")

    frames = [read_source_csv(path, source) for path, source in zip(input_csvs, source_names)]
    all_columns = []
    for frame in frames:
        for col in frame.columns:
            if col not in all_columns:
                all_columns.append(col)

    frames = [frame.reindex(columns=all_columns, fill_value="") for frame in frames]
    combined = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"[DONE] Wrote combined clinical-state CSV: {output_csv}")
    for source, frame in zip(source_names, frames):
        print(f"[INFO] {source}: {len(frame):,} rows")
    print(f"[INFO] total: {len(combined):,} rows")


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Combine multiple clinical-state CSVs for RAG.")
    parser.add_argument("--input_csvs", nargs="+", required=True)
    parser.add_argument("--source_names", nargs="+", required=True)
    parser.add_argument("--output_csv", default=str(root / "data" / "combined_train_clinical_state.csv"))
    args = parser.parse_args()
    combine([Path(path) for path in args.input_csvs], args.source_names, Path(args.output_csv))


if __name__ == "__main__":
    main()

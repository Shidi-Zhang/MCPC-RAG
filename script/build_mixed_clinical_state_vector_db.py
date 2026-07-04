#!/usr/bin/env python3
"""
Build a FAISS vector database for a mixed MIMIC-IV + eICU clinical-state CSV.

The metadata keeps source/hospital fields when they exist, which makes later
debugging easier. Existing vector DBs are not touched unless the same output
directory is explicitly supplied.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import pandas as pd
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: Path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        resolved_model_path = model_path.resolve()
        try:
            self.model = SentenceTransformer(str(resolved_model_path), local_files_only=True)
        except TypeError:
            self.model = SentenceTransformer(str(resolved_model_path))

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def safe_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in text.split(",") if p.strip()]


def build_vector_db(input_csv: Path, output_dir: Path, text_col: str, model_path: Path, index_name: str) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    required = {"subject_id", "hadm_id", "diagnoses", "medications", text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    texts = []
    metadatas = []
    for row in df.to_dict("records"):
        diagnoses = safe_list(row.get("diagnoses"))
        medications = safe_list(row.get("medications"))
        procedures = safe_list(row.get("procedures")) if "procedures" in row else []
        lab_summary = safe_list(row.get("lab_summary")) if "lab_summary" in row else []
        vital_summary = safe_list(row.get("vital_summary")) if "vital_summary" in row else []
        text = str(row.get(text_col) or "").strip()
        if not text:
            text = f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}"
        texts.append(text)
        metadatas.append(
            {
                "source": str(row.get("source", "")),
                "subject_id": str(row.get("subject_id", "")),
                "hadm_id": str(row.get("hadm_id", "")),
                "hospitalid": str(row.get("hospitalid", "")),
                "diagnoses": diagnoses,
                "medications": medications,
                "procedures": procedures,
                "lab_summary": lab_summary,
                "vital_summary": vital_summary,
                "clinical_state": text,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = SentenceTransformerEmbeddings(model_path)
    store = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    store.save_local(str(output_dir), index_name)
    print(f"[DONE] Built mixed clinical-state vector DB at: {output_dir}")
    print(f"[INFO] index_name: {index_name}")
    print(f"[INFO] rows: {len(texts):,}")


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build mixed MIMIC/eICU clinical-state FAISS vector DB.")
    parser.add_argument("--input_csv", default=str(root / "data" / "mimic_eicu_train_clinical_state.csv"))
    parser.add_argument("--output_dir", default=str(root / "vector_db" / "mimic_eicu_clinical_state_faiss_index"))
    parser.add_argument("--text_col", default="clinical_state")
    parser.add_argument("--model_path", default=str(root / "all-MiniLM-L6-v2"))
    parser.add_argument("--index_name", default="faiss_mimic_eicu_clinical_state_index")
    args = parser.parse_args()
    build_vector_db(
        Path(args.input_csv),
        Path(args.output_dir),
        args.text_col,
        Path(args.model_path),
        args.index_name,
    )


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataSpec:
    train_csv: str | None
    test_csv: str
    code_col: str
    label_col: str
    graph_col: str

def _read_csv_auto(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".zst"):
        return pd.read_csv(p, compression="zstd")
    return pd.read_csv(p, compression="infer")

def load_clean_csv(csv_path: str, code_col: str, label_col: str, graph_col: str) -> pd.DataFrame:
    df = _read_csv_auto(csv_path)

    need = [code_col, label_col, graph_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{csv_path} missing column: {c} (has={list(df.columns)[:20]}...)")

    df = df.dropna(subset=need).reset_index(drop=True)
    mask_empty = df[graph_col].astype(str).str.strip().eq("")
    if mask_empty.any():
        df = df.loc[~mask_empty].reset_index(drop=True)

    df[label_col] = df[label_col].astype(int)
    df["label"] = df[label_col].astype(int)  # unified
    return df

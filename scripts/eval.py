# scripts/eval.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    roc_auc_score, confusion_matrix
)
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.data import load_clean_csv
from src.graph_features import FEATURE_DIM, features_from_graphson_str
from src.model import Qwen2p5CoderQLoRAWithGraph


class CodeSymDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, code_col, graph_col, cache_dir):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.code_col = code_col
        self.graph_col = graph_col
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row[self.code_col])

        toks = self.tok(
            code,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        gvec = features_from_graphson_str(str(row[self.graph_col]), self.cache_dir)

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": torch.tensor(int(row["label"]), dtype=torch.float),
            "graph_feats": torch.tensor(gvec, dtype=torch.float),
        }


class SymCollator(DataCollatorWithPadding):
    def __call__(self, features):
        graph_feats = torch.stack([f.pop("graph_feats") for f in features], dim=0)
        labels = torch.stack([f.pop("labels") for f in features], dim=0)
        batch = super().__call__(features)
        batch["graph_feats"] = graph_feats
        batch["labels"] = labels
        return batch


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray):
    logits = np.array(logits).reshape(-1)
    labels = np.array(labels).reshape(-1).astype(int)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-12)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(specificity),
        "mcc": float(mcc),
        "auc": float(auc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


@torch.no_grad()
def run_eval(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    model_name = cfg["model_name"]
    bf16 = bool(cfg.get("bf16", True))
    max_len = int(cfg.get("max_len", 512))
    batch_eval = int(cfg.get("batch_eval", 8))

    test_csv = str(ROOT / cfg["data"]["test_csv"])
    code_col = cfg["data"]["code_col"]
    label_col = cfg["data"]["label_col"]
    graph_col = cfg["data"]["graph_col"]

    cache_dir = str(ROOT / cfg["cache"]["dir"])
    best_dir = str(ROOT / cfg["artifacts"]["best_dir"])
    out_json = str(ROOT / cfg["output"]["metrics_json"])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[paths] best_dir={best_dir}")
    print(f"[paths] test_csv={test_csv}")


    df = load_clean_csv(test_csv, code_col, label_col, graph_col)
    print(f"[data] test={len(df)}")


    try:
        tok = AutoTokenizer.from_pretrained(best_dir, use_fast=False, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = CodeSymDataset(df, tok, max_len, code_col, graph_col, cache_dir)
    collator = SymCollator(tok)
    dl = DataLoader(ds, batch_size=batch_eval, shuffle=False, collate_fn=collator)


    lora_cfg = cfg.get("lora", {"r": 16, "alpha": 32, "dropout": 0.05})
    model = Qwen2p5CoderQLoRAWithGraph.from_artifacts(
        model_name=model_name,
        artifact_dir=best_dir,
        bf16=bf16,
        graph_feat_dim=FEATURE_DIM,
        dropout=float(cfg.get("dropout", 0.1)),
        lora_cfg=lora_cfg,
        device=device,
    )


    all_logits = []
    all_labels = []

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        gfeats = batch["graph_feats"].to(device)
        labels = batch["labels"].cpu().numpy()

        out = model(input_ids=input_ids, attention_mask=attn, graph_feats=gfeats, labels=None)
        logits = out["logits"].detach().float().cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics_from_logits(all_logits, all_labels)
    print("[eval] metrics:", metrics)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[done] saved metrics ->", out_json)


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to eval.yaml")
    args = ap.parse_args()
    main(args.config)

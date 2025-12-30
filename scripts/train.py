# scripts/train.py
import os, json
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorWithPadding, set_seed
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.data import load_clean_csv
from src.graph_features import FEATURE_DIM, features_from_graphson_str
from src.model import Qwen2p5CoderQLoRAWithGraph, build_bnb_config, build_lora_config


class CodeSymDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, code_col, graph_col, cache_dir):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
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
        labels = torch.stack([f["labels"] for f in features], dim=0)
        batch = super().__call__(features)
        batch["graph_feats"] = graph_feats
        batch["labels"] = labels
        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.array(logits).reshape(-1)
    labels = np.array(labels).reshape(-1).astype(int)

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-12)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "mcc": mcc,
        "auc": auc,
    }


class SymTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            graph_feats=inputs["graph_feats"],
            labels=inputs.get("labels"),
            pos_weight=self._pos_weight.to(inputs["input_ids"].device),
        )
        loss = out["loss"]
        return (loss, out) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                graph_feats=inputs["graph_feats"],
                labels=None,
                pos_weight=None,
            )
        return (None, out["logits"], inputs.get("labels"))


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    set_seed(int(cfg["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = cfg["model_name"]
    max_len = int(cfg["max_len"])

    train_csv = str(ROOT / cfg["data"]["train_csv"])
    test_csv  = str(ROOT / cfg["data"]["test_csv"])
    code_col  = cfg["data"]["code_col"]
    label_col = cfg["data"]["label_col"]
    graph_col = cfg["data"]["graph_col"]

    cache_dir = str(ROOT / cfg["cache"]["dir"])
    out_dir   = str(ROOT / cfg["output"]["dir"])
    best_dir  = os.path.join(out_dir, "best_model")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ---- data ----
    train_df = load_clean_csv(train_csv, code_col, label_col, graph_col)
    test_df  = load_clean_csv(test_csv,  code_col, label_col, graph_col)
    print(f"[data] train={len(train_df)} test={len(test_df)}")

    # pos_weight
    y = train_df["label"].values.astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    pw = float(neg) / max(1.0, float(pos))
    pos_weight = torch.tensor([pw], dtype=torch.float)
    print(f"[class-balance] pos_weight={pw:.4f} (neg={neg}, pos={pos})")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = CodeSymDataset(train_df, tok, max_len, code_col, graph_col, cache_dir)
    test_ds  = CodeSymDataset(test_df,  tok, max_len, code_col, graph_col, cache_dir)
    collator = SymCollator(tok)

    # ---- model ----
    bf16 = bool(cfg["train"]["bf16"])
    bnb = build_bnb_config(bf16=bf16)
    lora = build_lora_config(
        r=int(cfg["lora"]["r"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
    )

    model = Qwen2p5CoderQLoRAWithGraph(
        model_name=model_name,
        bnb_config=bnb,
        lora_config=lora,
        graph_feat_dim=FEATURE_DIM,
        bf16=bf16,
        dropout=float(cfg["train"].get("dropout", 0.1)),
    )


    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=int(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"]["lr"]),
        per_device_train_batch_size=int(cfg["train"]["batch_train"]),
        per_device_eval_batch_size=int(cfg["train"]["batch_eval"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=int(cfg["train"].get("logging_steps", 50)),
        save_total_limit=int(cfg["train"].get("save_total_limit", 2)),
        report_to="none",
        seed=int(cfg["seed"]),

        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        warmup_ratio=float(cfg["train"].get("warmup_ratio", 0.0)),

        fp16=bool(cfg["train"]["fp16"]),
        bf16=bool(cfg["train"]["bf16"]),
        gradient_checkpointing=bool(cfg["train"]["grad_ckpt"]),
        save_safetensors=False,

        # keep graph_feats in batch
        remove_unused_columns=False,
    )

    trainer = SymTrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # store config copy
    os.makedirs(best_dir, exist_ok=True)
    with open(os.path.join(best_dir, "train_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("[training] start...")
    trainer.train()
    print("[training] done.")

    print("[test] evaluating...")
    metrics = trainer.evaluate(test_ds)
    print("[test] metrics:", metrics)

    # save artifacts
    tok.save_pretrained(best_dir)
    model.save_artifacts(best_dir)

    with open(os.path.join(best_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print("[done] saved ->", best_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to train.yaml")
    args = ap.parse_args()
    main(args.config)

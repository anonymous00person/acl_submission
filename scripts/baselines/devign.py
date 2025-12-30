# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as f

from gensim.models import Word2Vec

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)


def load_yaml(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


token_re = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||[^\s]")


def code_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return token_re.findall(s)


def extract_method_name(code: str) -> Optional[str]:
    if not isinstance(code, str) or not code.strip():
        return None
    first = code.strip().splitlines()[0]
    m = re.search(r"\b([A-Za-z_]\w*)\s*\(", first)
    if not m:
        return None
    name = m.group(1)
    if name in {"if", "for", "while", "switch", "catch", "return", "new"}:
        return None
    return name


def gval(x: Any) -> Any:
    if isinstance(x, dict) and "@value" in x:
        return gval(x["@value"])
    return x


def prop_first(props: Dict[str, Any], key: str, default=None):
    if not isinstance(props, dict) or key not in props:
        return default
    node = props.get(key, {})
    node = node.get("@value", {})
    node = node.get("@value", [])
    if isinstance(node, list) and node:
        return gval(node[0])
    return default


@dataclass
class parsed_graph:
    vertices: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


def parse_graphson(graphson_str: str) -> Optional[parsed_graph]:
    try:
        obj = json.loads(graphson_str)
    except Exception:
        return None

    g = obj.get("@value", obj)
    if isinstance(g, dict) and "vertices" not in g and "@value" in g:
        g = g.get("@value", g)

    if not isinstance(g, dict):
        return None

    vertices = g.get("vertices", [])
    edges = g.get("edges", [])
    if not isinstance(vertices, list) or not isinstance(edges, list):
        return None
    return parsed_graph(vertices=vertices, edges=edges)


ast_rel_labels = {"AST", "CONTAINS"}
cfg_labels = {"CFG"}
dfg_labels = {"REACHING_DEF", "DDG", "DATA_DEPENDENCY"}
rel_names = ["AST", "CFG", "DFG", "NCS"]


def build_edge_index(pg: parsed_graph):
    id2pos: Dict[Any, int] = {}
    for i, v in enumerate(pg.vertices):
        vid = gval(v.get("id"))
        id2pos[vid] = i

    eds: List[Tuple[Any, Any, str]] = []
    for e in pg.edges:
        outv = gval(e.get("outV"))
        inv = gval(e.get("inV"))
        lab = e.get("label")
        if outv in id2pos and inv in id2pos and isinstance(lab, str):
            eds.append((outv, inv, lab))
    return id2pos, eds


def pick_target_method_id(pg: parsed_graph, row_line: Optional[int], func_code: str):
    cands: List[Tuple[Any, Any, Any, str]] = []
    for v in pg.vertices:
        if v.get("label") != "METHOD":
            continue
        props = v.get("properties", {})
        is_ext = prop_first(props, "IS_EXTERNAL", False)
        if is_ext in [True, "true", "True", 1]:
            continue
        vid = gval(v.get("id"))
        line_no = prop_first(props, "LINE_NUMBER", None)
        name = prop_first(props, "NAME", None)
        code = prop_first(props, "CODE", "") or ""
        cands.append((vid, line_no, name, code))

    if not cands:
        return None

    if row_line is not None and not (isinstance(row_line, float) and math.isnan(row_line)):
        try:
            rl = int(row_line)
            line_matches = [
                c for c in cands
                if isinstance(c[1], (int, np.integer)) and int(c[1]) == rl
            ]
            if line_matches:
                return line_matches[0][0]
        except Exception:
            pass

    mname = extract_method_name(func_code)
    if mname:
        name_matches = [c for c in cands if isinstance(c[2], str) and c[2] == mname]
        if name_matches:
            return name_matches[0][0]

    first_line = func_code.strip().splitlines()[0] if isinstance(func_code, str) and func_code.strip() else ""
    ftoks = set(code_tokens(first_line))

    best = None
    best_score = -1.0
    for vid, _, _, mcode in cands:
        mtoks = set(code_tokens(mcode))
        score = 0.0 if (not ftoks and not mtoks) else len(ftoks & mtoks) / (len(ftoks | mtoks) + 1e-9)
        if score > best_score:
            best_score = score
            best = vid
    return best


def extract_method_subgraph(pg: parsed_graph, method_id, max_nodes: int):
    _, all_edges = build_edge_index(pg)

    ast_adj: Dict[Any, List[Any]] = {}
    for u, v, lab in all_edges:
        if lab in ast_rel_labels:
            ast_adj.setdefault(u, []).append(v)

    visited: List[Any] = []
    seen = {method_id}
    q = [method_id]

    while q and len(visited) < max_nodes:
        u = q.pop(0)
        visited.append(u)
        for w in ast_adj.get(u, []):
            if w not in seen:
                seen.add(w)
                q.append(w)
            if len(visited) >= max_nodes:
                break

    node_ids = visited
    node_set = set(node_ids)

    kept: List[Tuple[Any, Any, str]] = []
    for u, v, lab in all_edges:
        if u in node_set and v in node_set:
            if lab in ast_rel_labels or lab in cfg_labels or lab in dfg_labels:
                kept.append((u, v, lab))

    return node_ids, kept


def compute_node_order(pg: parsed_graph, node_ids):
    id2pos = {gval(v.get("id")): i for i, v in enumerate(pg.vertices)}

    def key_fn(vid):
        v = pg.vertices[id2pos[vid]]
        props = v.get("properties", {})
        ln = prop_first(props, "LINE_NUMBER", 10**9)
        cn = prop_first(props, "COLUMN_NUMBER", 10**9)
        od = prop_first(props, "ORDER", 10**9)
        return (
            int(ln) if isinstance(ln, (int, np.integer)) else 10**9,
            int(cn) if isinstance(cn, (int, np.integer)) else 10**9,
            int(od) if isinstance(od, (int, np.integer)) else 10**9,
            id2pos.get(vid, 10**9),
        )

    return sorted(node_ids, key=key_fn)


def add_ncs_edges(ordered_node_ids, existing_edges):
    node_set = set(ordered_node_ids)
    out_ast = {vid: 0 for vid in node_set}
    for u, v, lab in existing_edges:
        if lab in ast_rel_labels and u in node_set and v in node_set:
            out_ast[u] += 1
    leaves = [vid for vid in ordered_node_ids if out_ast.get(vid, 0) == 0]
    ncs = [(a, b, "NCS") for a, b in zip(leaves, leaves[1:])]
    return existing_edges + ncs


def train_w2v(train_df: pd.DataFrame, code_col: str, w2v_dim: int) -> Word2Vec:
    sents: List[List[str]] = []
    for s in tqdm(train_df[code_col].astype(str).tolist(), desc="w2v"):
        toks = code_tokens(s)
        if toks:
            sents.append(toks)

    return Word2Vec(
        sentences=sents,
        vector_size=w2v_dim,
        window=5,
        min_count=1,
        workers=max(1, os.cpu_count() or 1),
        sg=1,
        epochs=5,
    )


def w2v_avg(w2v: Word2Vec, text: str, w2v_dim: int):
    toks = code_tokens(text)
    if not toks:
        return np.zeros((w2v_dim,), dtype=np.float32)
    vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
    if not vecs:
        return np.zeros((w2v_dim,), dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def build_type_vocab(train_df: pd.DataFrame, g_col: str):
    vocab: Dict[str, int] = {"<unk_type>": 0}
    for gs in tqdm(train_df[g_col].astype(str).tolist(), desc="types"):
        pg = parse_graphson(gs)
        if pg is None:
            continue
        for v in pg.vertices:
            lab = v.get("label")
            if isinstance(lab, str) and lab not in vocab:
                vocab[lab] = len(vocab)
    return vocab


def row_to_sample(
    row: pd.Series,
    code_col: str,
    label_col: str,
    g_col: str,
    line_col: Optional[str],
    w2v: Word2Vec,
    type_vocab: Dict[str, int],
    w2v_dim: int,
    max_nodes: int,
):
    gs = row[g_col]
    pg = parse_graphson(gs)
    if pg is None:
        return None

    func_code = row[code_col]
    row_line = row[line_col] if line_col else None

    method_id = pick_target_method_id(pg, row_line, func_code)
    if method_id is None:
        return None

    node_ids, edges = extract_method_subgraph(pg, method_id, max_nodes=max_nodes)
    if len(node_ids) < 2:
        return None

    ordered_ids = compute_node_order(pg, node_ids)
    edges = add_ncs_edges(ordered_ids, edges)

    id2new = {vid: i for i, vid in enumerate(ordered_ids)}
    id2pos = {gval(v.get("id")): i for i, v in enumerate(pg.vertices)}

    x_code = np.zeros((len(ordered_ids), w2v_dim), dtype=np.float32)
    type_ids = np.zeros((len(ordered_ids),), dtype=np.int64)

    for i, vid in enumerate(ordered_ids):
        v = pg.vertices[id2pos[vid]]
        vlab = v.get("label") if isinstance(v.get("label"), str) else "<unk_type>"
        type_ids[i] = type_vocab.get(vlab, 0)
        props = v.get("properties", {})
        code_txt = prop_first(props, "CODE", "") or ""
        if code_txt == "<empty>":
            code_txt = ""
        x_code[i] = w2v_avg(w2v, code_txt, w2v_dim)

    edges_by_rel: Dict[str, List[Tuple[int, int]]] = {k: [] for k in rel_names}
    for u, v, lab in edges:
        if u not in id2new or v not in id2new:
            continue
        if lab in ast_rel_labels:
            edges_by_rel["AST"].append((id2new[u], id2new[v]))
        elif lab in cfg_labels:
            edges_by_rel["CFG"].append((id2new[u], id2new[v]))
        elif lab in dfg_labels:
            edges_by_rel["DFG"].append((id2new[u], id2new[v]))
        elif lab == "NCS":
            edges_by_rel["NCS"].append((id2new[u], id2new[v]))

    edge_tensors: Dict[str, torch.Tensor] = {}
    for rel in rel_names:
        pairs = edges_by_rel[rel]
        if not pairs:
            edge_tensors[rel] = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_tensors[rel] = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    y = int(row[label_col])

    return {
        "x_code": torch.tensor(x_code, dtype=torch.float16),
        "type_ids": torch.tensor(type_ids, dtype=torch.long),
        "edges": edge_tensors,
        "y": y,
    }


def build_cache(
    df: pd.DataFrame,
    cache_path: str,
    code_col: str,
    label_col: str,
    g_col: str,
    line_col: Optional[str],
    w2v: Word2Vec,
    type_vocab: Dict[str, int],
    w2v_dim: int,
    max_nodes: int,
):
    if os.path.exists(cache_path):
        return cache_path

    samples = []
    bad = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=Path(cache_path).name):
        s = row_to_sample(row, code_col, label_col, g_col, line_col, w2v, type_vocab, w2v_dim, max_nodes)
        if s is None:
            bad += 1
            continue
        samples.append(s)

    ensure_dir(str(Path(cache_path).parent))
    torch.save({"samples": samples, "bad": bad}, cache_path)
    return cache_path


class graph_dataset(torch.utils.data.Dataset):
    def __init__(self, cache_path: str):
        obj = torch.load(cache_path, map_location="cpu")
        self.samples = obj["samples"]
        self.bad = obj.get("bad", 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_graphs(batch):
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    return batch, ys


class devign(nn.Module):
    def __init__(
        self,
        num_types: int,
        w2v_dim: int,
        hid_dim: int,
        time_steps: int,
        max_nodes: int,
        type_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.w2v_dim = w2v_dim
        self.hid_dim = hid_dim
        self.time_steps = time_steps
        self.max_nodes = max_nodes
        self.device = device

        self.type_emb = nn.Embedding(num_types, type_dim)
        self.in_proj = nn.Linear(w2v_dim + type_dim, hid_dim)

        self.rel_lin = nn.ModuleDict({k: nn.Linear(hid_dim, hid_dim, bias=True) for k in rel_names})
        self.gru = nn.GRUCell(hid_dim, hid_dim)

        conv_in = 2 * hid_dim
        self.conv1 = nn.Conv1d(in_channels=conv_in, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros((1, conv_in, max_nodes))
            z = self.pool2(f.relu(self.conv2(self.pool1(f.relu(self.conv1(dummy))))))
            flat_dim = z.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def ggnn(self, x0: torch.Tensor, edges: Dict[str, torch.Tensor]):
        h = x0
        n = h.size(0)
        for _ in range(self.time_steps):
            m = torch.zeros((n, self.hid_dim), device=h.device, dtype=h.dtype)
            for rel in rel_names:
                e = edges[rel]
                if e.numel() == 0:
                    continue
                src = e[0].to(h.device)
                dst = e[1].to(h.device)
                msg = self.rel_lin[rel](h[src])
                m.index_add_(0, dst, msg)
            h = self.gru(m, h)
        return h

    def forward_one(self, sample: Dict[str, Any]):
        x_code = sample["x_code"].to(self.device).float()
        type_ids = sample["type_ids"].to(self.device)
        edges = {k: v.to(self.device) for k, v in sample["edges"].items()}

        t = self.type_emb(type_ids)
        x_init = self.in_proj(torch.cat([x_code, t], dim=1))

        ht = self.ggnn(x_init, edges)
        x = torch.cat([x_init, ht], dim=1)

        n = x.size(0)
        if n < self.max_nodes:
            pad = torch.zeros((self.max_nodes - n, x.size(1)), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        elif n > self.max_nodes:
            x = x[: self.max_nodes]

        z = x.t().unsqueeze(0)
        z = f.relu(self.conv1(z))
        z = self.pool1(z)
        z = f.relu(self.conv2(z))
        z = self.pool2(z)

        z = z.view(1, -1)
        z = f.relu(self.fc1(z))
        logits = self.fc2(z)
        return logits

    def forward(self, batch_samples: List[Dict[str, Any]]):
        outs = [self.forward_one(s) for s in batch_samples]
        return torch.cat(outs, dim=0)


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1v, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-9)

    return {
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1v),
        "specificity": float(spec),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "auc": float(auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_epoch(model: devign, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    all_y: List[int] = []
    all_prob: List[float] = []
    total_loss = 0.0
    n = 0

    for batch_samples, ys in tqdm(loader, desc=("train" if train_mode else "eval"), leave=False):
        ys = ys.to(model.device)
        logits = model(batch_samples)
        loss = f.cross_entropy(logits, ys)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_prob.extend(probs.tolist())
        all_y.extend(ys.detach().cpu().numpy().tolist())

        total_loss += float(loss.item()) * len(batch_samples)
        n += len(batch_samples)

    metrics = compute_metrics(np.array(all_y), np.array(all_prob))
    metrics["loss"] = float(total_loss / max(1, n))
    return metrics, np.array(all_y), np.array(all_prob)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_s = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    out_dir = str(root / cfg["output"]["dir"])
    ensure_dir(out_dir)

    train_csv = str(root / cfg["data"]["train_csv"])
    test_csv = str(root / cfg["data"]["test_csv"])
    code_col = cfg["data"]["code_col"]
    label_col = cfg["data"]["label_col"]
    g_col = cfg["data"]["graph_col"]
    line_col = cfg["data"].get("line_col", None)

    w2v_dim = int(cfg["model"]["w2v_dim"])
    hid_dim = int(cfg["model"]["hid_dim"])
    time_steps = int(cfg["model"]["time_steps"])
    max_nodes = int(cfg["model"]["max_nodes"])
    type_dim = int(cfg["model"]["type_dim"])

    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    patience_limit = int(cfg["train"]["early_stop_patience"])

    w2v_path = str(root / cfg["output"]["w2v_path"])
    type_vocab_json = str(root / cfg["output"]["type_vocab_json"])
    train_cache_pt = str(root / cfg["output"]["train_cache_pt"])
    test_cache_pt = str(root / cfg["output"]["test_cache_pt"])
    best_ckpt_pt = str(root / cfg["output"]["best_ckpt_pt"])
    preds_csv = str(root / cfg["output"]["preds_csv"])
    metrics_json = str(root / cfg["output"]["metrics_json"])

    for p in [w2v_path, type_vocab_json, train_cache_pt, best_ckpt_pt, preds_csv, metrics_json]:
        ensure_dir(str(Path(p).parent))

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    for c in [code_col, label_col, g_col]:
        if c not in df_tr.columns:
            raise KeyError(f"train missing column: {c}")
        if c not in df_te.columns:
            raise KeyError(f"test missing column: {c}")

    if line_col and line_col not in df_tr.columns:
        line_col = None
    if line_col and line_col not in df_te.columns:
        line_col = None

    if os.path.exists(w2v_path):
        w2v = Word2Vec.load(w2v_path)
    else:
        w2v = train_w2v(df_tr, code_col, w2v_dim)
        w2v.save(w2v_path)

    if os.path.exists(type_vocab_json):
        with open(type_vocab_json, "r", encoding="utf-8") as fobj:
            type_vocab = json.load(fobj)
    else:
        type_vocab = build_type_vocab(df_tr, g_col)
        with open(type_vocab_json, "w", encoding="utf-8") as fobj:
            json.dump(type_vocab, fobj, indent=2)

    build_cache(df_tr, train_cache_pt, code_col, label_col, g_col, line_col, w2v, type_vocab, w2v_dim, max_nodes)
    build_cache(df_te, test_cache_pt, code_col, label_col, g_col, line_col, w2v, type_vocab, w2v_dim, max_nodes)

    ds_train = graph_dataset(train_cache_pt)
    ds_test = graph_dataset(test_cache_pt)

    print(f"[env] device={device} seed={seed}")
    print(f"[data] train={len(ds_train)} skipped_train={ds_train.bad} test={len(ds_test)} skipped_test={ds_test.bad}")

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs, num_workers=0
    )

    model = devign(
        num_types=len(type_vocab),
        w2v_dim=w2v_dim,
        hid_dim=hid_dim,
        time_steps=time_steps,
        max_nodes=max_nodes,
        type_dim=type_dim,
        device=device,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    patience = 0

    for epoch in range(1, epochs + 1):
        tr_metrics, _, _ = run_epoch(model, train_loader, optimizer=opt)
        te_metrics, y_true, y_prob = run_epoch(model, test_loader, optimizer=None)

        print(
            f"epoch {epoch:03d} "
            f"train loss={tr_metrics['loss']:.4f} f1={tr_metrics['f1']:.4f} auc={tr_metrics['auc']:.4f} "
            f"test loss={te_metrics['loss']:.4f} f1={te_metrics['f1']:.4f} auc={te_metrics['auc']:.4f} "
            f"acc={te_metrics['acc']:.4f} mcc={te_metrics['mcc']:.4f}"
        )

        if te_metrics["f1"] > best_f1:
            best_f1 = float(te_metrics["f1"])
            patience = 0
            torch.save({"model": model.state_dict(), "type_vocab": type_vocab, "cfg": cfg}, best_ckpt_pt)
        else:
            patience += 1
            if patience >= patience_limit:
                break

    ckpt = torch.load(best_ckpt_pt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    te_metrics, y_true, y_prob = run_epoch(model, test_loader, optimizer=None)

    with open(metrics_json, "w", encoding="utf-8") as fobj:
        json.dump({"test": te_metrics, "seed": seed, "best_f1": best_f1}, fobj, indent=2)

    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": (y_prob >= 0.5).astype(int)}).to_csv(preds_csv, index=False)

    print(
        f"[test] acc={te_metrics['acc']:.4f} f1={te_metrics['f1']:.4f} "
        f"mcc={te_metrics['mcc']:.4f} auc={te_metrics['auc']:.4f} "
        f"precision={te_metrics['precision']:.4f} recall={te_metrics['recall']:.4f} "
        f"specificity={te_metrics['specificity']:.4f} kappa={te_metrics['kappa']:.4f}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)

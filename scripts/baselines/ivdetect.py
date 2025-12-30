# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as f

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_any(path: str) -> pd.DataFrame:
    return pd.read_csv(str(path))


def unwrap_gvalue(x):
    if isinstance(x, dict) and "@type" in x and "@value" in x:
        return unwrap_gvalue(x["@value"])
    return x


def get_vprop(v: dict, key: str, default=None):
    props = v.get("properties", {})
    if key not in props:
        return default
    vp = props[key]
    try:
        val = vp.get("@value", None)
        if val is None:
            return default
        if isinstance(val, dict) and "@value" in val:
            lst = val["@value"]
        else:
            lst = val
        if isinstance(lst, list) and len(lst) > 0:
            return unwrap_gvalue(lst[0])
        return unwrap_gvalue(lst)
    except Exception:
        return default


def parse_graphson(graphson_str: str):
    g = json.loads(graphson_str)
    gv = g.get("@value", {})
    vertices = gv.get("vertices", []) or []
    edges = gv.get("edges", []) or []
    out_vertices = []
    for v in vertices:
        if isinstance(v, dict):
            out_vertices.append(v)
    return out_vertices, edges


def eid_out_in(e: dict) -> Tuple[Optional[int], Optional[int], str]:
    lbl = e.get("label", "") or ""
    outv = unwrap_gvalue(e.get("outV", None))
    inv = unwrap_gvalue(e.get("inV", None))
    try:
        outv = int(outv) if outv is not None else None
    except Exception:
        outv = None
    try:
        inv = int(inv) if inv is not None else None
    except Exception:
        inv = None
    return outv, inv, lbl


java_keywords = set(
    """
abstract assert boolean break byte case catch char class const continue default do double else enum extends
final finally float for goto if implements import instanceof int interface long native new package private
protected public return short static strictfp super switch synchronized this throw throws transient try
void volatile while true false null
""".split()
)

ident_pat = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_camel(s: str) -> List[str]:
    parts = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s).split()
    out = []
    for p in parts:
        out.extend(re.split(r"[_$]+", p))
    return [x for x in out if x]


def subtokenize(code: str, max_tokens: int) -> List[str]:
    if not isinstance(code, str):
        return []
    toks = ident_pat.findall(code)
    subtoks = []
    for t in toks:
        if t in java_keywords:
            continue
        for st in split_camel(t):
            st = st.strip()
            if len(st) <= 1:
                continue
            subtoks.append(st.lower())
            if len(subtoks) >= max_tokens:
                return subtoks
    return subtoks[:max_tokens]


def hash_token(tok: str, mod: int) -> int:
    h = hashlib.md5(tok.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % mod


pdg_edge_labels = {
    "CFG",
    "CDG",
    "REACHING_DEF",
    "DOMINATE",
    "POST_DOMINATE",
    "CONTROL_DEPENDENCE",
    "DATA_DEPENDENCE",
    "DDG",
}
ast_edge_label = "AST"

var_like_labels = {
    "IDENTIFIER",
    "LOCAL",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "FIELD_IDENTIFIER",
}


@dataclass
class statement_pack:
    line: int
    stmt_tokens: List[int]
    varname_tokens: List[int]
    vartype_tokens: List[int]
    ctx_neighbors: List[int]
    ast_nodes: List[int]
    ast_children: Dict[int, List[int]]
    ast_node_label_ids: Dict[int, int]
    ast_node_tok_ids: Dict[int, List[int]]


@dataclass
class graph_example:
    y: int
    stmts: List[statement_pack]
    edges: List[Tuple[int, int]]


def build_graph_example(
    graphson_str: str,
    y: int,
    token_vocab: int,
    label_vocab: int,
    max_stmt_tokens: int,
    max_var_tokens: int,
    max_ctx_nei: int,
    max_ast_nodes: int,
    max_ast_depth: int,
) -> Optional[graph_example]:
    try:
        vertices, edges = parse_graphson(graphson_str)
    except Exception:
        return None

    vid2v: Dict[int, dict] = {}
    for v in vertices:
        vv_id = unwrap_gvalue(v.get("id", None))
        if vv_id is None:
            continue
        try:
            vid2v[int(vv_id)] = v
        except Exception:
            continue

    line2vids: Dict[int, List[int]] = {}
    for vv_id, v in vid2v.items():
        ln = get_vprop(v, "LINE_NUMBER", None)
        if ln is None:
            continue
        try:
            ln = int(ln)
        except Exception:
            continue
        line2vids.setdefault(ln, []).append(vv_id)

    if not line2vids:
        return None

    lines = sorted(line2vids.keys())
    line2sidx = {ln: i for i, ln in enumerate(lines)}

    def collect_stmt_code(ln: int) -> str:
        codes = []
        for vv_id in line2vids.get(ln, []):
            v = vid2v[vv_id]
            c = get_vprop(v, "CODE", None)
            if isinstance(c, str) and c.strip():
                codes.append(c.strip())
        if not codes:
            for vv_id in line2vids.get(ln, []):
                v = vid2v[vv_id]
                n = get_vprop(v, "NAME", None)
                if isinstance(n, str) and n.strip():
                    codes.append(n.strip())
        return " ".join(codes)[:2000]

    def collect_vars_types(ln: int) -> Tuple[str, str]:
        varnames, vartypes = [], []
        for vv_id in line2vids.get(ln, []):
            v = vid2v[vv_id]
            lbl = v.get("label", "") or ""
            if lbl in var_like_labels:
                nm = get_vprop(v, "NAME", None)
                cd = get_vprop(v, "CODE", None)
                if isinstance(nm, str) and nm.strip():
                    varnames.append(nm.strip())
                elif isinstance(cd, str) and cd.strip():
                    varnames.append(cd.strip())
                t = get_vprop(v, "TYPE_FULL_NAME", None)
                if isinstance(t, str) and t.strip():
                    vartypes.append(t.strip())
        return " ".join(varnames)[:2000], " ".join(vartypes)[:2000]

    ast_children_global: Dict[int, List[int]] = {}
    for e in edges:
        outv, inv, lbl = eid_out_in(e)
        if lbl != ast_edge_label:
            continue
        if outv is None or inv is None:
            continue
        ast_children_global.setdefault(outv, []).append(inv)

    def choose_ast_root(ln: int) -> Optional[int]:
        cands = []
        for vv_id in line2vids.get(ln, []):
            v = vid2v[vv_id]
            order = get_vprop(v, "ORDER", 10**9)
            lbl = v.get("label", "") or ""
            penalty = 0
            if lbl in ("BLOCK", "METHOD", "TYPE_DECL", "NAMESPACE_BLOCK", "FILE", "TYPE"):
                penalty = 1000
            try:
                order = int(order) if order is not None else 10**9
            except Exception:
                order = 10**9
            cands.append((penalty, order, vv_id))
        cands.sort()
        return cands[0][2] if cands else None

    def extract_ast_subtree(ln: int) -> Tuple[List[int], Dict[int, List[int]]]:
        root_id = choose_ast_root(ln)
        if root_id is None:
            return [], {}
        nodes, children = [], {}
        q = [(root_id, 0)]
        seen = set()
        while q and len(nodes) < max_ast_nodes:
            cur, d = q.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            v = vid2v.get(cur, None)
            if v is None:
                continue
            ln_cur = get_vprop(v, "LINE_NUMBER", None)
            if ln_cur is not None:
                try:
                    ln_cur = int(ln_cur)
                except Exception:
                    ln_cur = None
            if ln_cur is not None and ln_cur != ln:
                continue
            nodes.append(cur)
            if d >= max_ast_depth:
                continue
            ch = ast_children_global.get(cur, [])
            kept = []
            for c in ch:
                if c in seen:
                    continue
                vv = vid2v.get(c, None)
                if vv is None:
                    continue
                ln_c = get_vprop(vv, "LINE_NUMBER", None)
                if ln_c is not None:
                    try:
                        ln_c = int(ln_c)
                    except Exception:
                        ln_c = None
                if ln_c is not None and ln_c != ln:
                    continue
                kept.append(c)
                q.append((c, d + 1))
            if kept:
                children[cur] = kept
        return nodes, children

    stmt_edges = set()
    for e in edges:
        outv, inv, lbl = eid_out_in(e)
        if outv is None or inv is None:
            continue
        if lbl not in pdg_edge_labels and lbl != "CFG":
            continue
        vo = vid2v.get(outv, None)
        vi = vid2v.get(inv, None)
        if vo is None or vi is None:
            continue
        lno = get_vprop(vo, "LINE_NUMBER", None)
        lni = get_vprop(vi, "LINE_NUMBER", None)
        if lno is None or lni is None:
            continue
        try:
            lno = int(lno)
            lni = int(lni)
        except Exception:
            continue
        if lno not in line2sidx or lni not in line2sidx:
            continue
        a = line2sidx[lno]
        b = line2sidx[lni]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        stmt_edges.add((a, b))

    stmt_edges = list(stmt_edges)
    neighbors = {i: set() for i in range(len(lines))}
    for a, b in stmt_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    stmts: List[statement_pack] = []
    for sidx, ln in enumerate(lines):
        stmt_code = collect_stmt_code(ln)
        vnames, vtypes = collect_vars_types(ln)

        stoks = [hash_token(t, token_vocab) for t in subtokenize(stmt_code, max_stmt_tokens)]
        vtoks = [hash_token(t, token_vocab) for t in subtokenize(vnames, max_var_tokens)]
        ttoks = [hash_token(t, token_vocab) for t in subtokenize(vtypes, max_var_tokens)]

        neis = sorted(list(neighbors.get(sidx, set())), key=lambda j: abs(lines[j] - ln))[:max_ctx_nei]

        ast_nodes, ast_children = extract_ast_subtree(ln)
        ast_node_label_ids = {}
        ast_node_tok_ids = {}
        for nid in ast_nodes:
            v = vid2v.get(nid, {})
            lbl = v.get("label", "UNK") or "UNK"
            ast_node_label_ids[nid] = hash_token(lbl, label_vocab)
            code_or_name = (get_vprop(v, "CODE", "") or "") + " " + (get_vprop(v, "NAME", "") or "")
            toks = [hash_token(t, token_vocab) for t in subtokenize(code_or_name, 16)]
            ast_node_tok_ids[nid] = toks

        stmts.append(
            statement_pack(
                line=ln,
                stmt_tokens=stoks,
                varname_tokens=vtoks,
                vartype_tokens=ttoks,
                ctx_neighbors=neis,
                ast_nodes=ast_nodes,
                ast_children=ast_children,
                ast_node_label_ids=ast_node_label_ids,
                ast_node_tok_ids=ast_node_tok_ids,
            )
        )

    if not stmts:
        return None
    return graph_example(y=int(y), stmts=stmts, edges=stmt_edges)


class attn_pool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        a = self.w(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=0)
        return (w.unsqueeze(-1) * x).sum(dim=0)


class seq_encoder(nn.Module):
    def __init__(self, vocab: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.pool = attn_pool(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: List[int]):
        if len(ids) == 0:
            return torch.zeros(self.emb.embedding_dim, device=self.emb.weight.device)
        x = torch.tensor(ids, dtype=torch.long, device=self.emb.weight.device).unsqueeze(0)
        e = self.emb(x)
        o, _ = self.gru(e)
        o = self.drop(o.squeeze(0))
        return self.pool(o)


class childsum_treelstm(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.w_iou = nn.Linear(in_dim, 3 * hid_dim)
        self.u_iou = nn.Linear(hid_dim, 3 * hid_dim, bias=False)
        self.w_f = nn.Linear(in_dim, hid_dim)
        self.u_f = nn.Linear(hid_dim, hid_dim, bias=False)

    def node_forward(self, x, child_h, child_c):
        if child_h is None or child_h.numel() == 0:
            h_sum = torch.zeros(self.u_iou.in_features, device=x.device)
        else:
            h_sum = child_h.sum(dim=0)
        iou = self.w_iou(x) + self.u_iou(h_sum)
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        if child_h is None or child_h.numel() == 0:
            c = i * u
        else:
            f_gate = torch.sigmoid(self.w_f(x).unsqueeze(0) + self.u_f(child_h))
            c = i * u + (f_gate * child_c).sum(dim=0)
        h = o * torch.tanh(c)
        return h, c


class ivd_like(nn.Module):
    def __init__(self, h: int, token_vocab: int, label_vocab: int, dropout: float = 0.1):
        super().__init__()
        self.h = h

        self.seq_stmt = seq_encoder(token_vocab, h, dropout=dropout)
        self.seq_varn = seq_encoder(token_vocab, h, dropout=dropout)
        self.seq_vart = seq_encoder(token_vocab, h, dropout=dropout)

        self.seq_ctx = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        self.ctx_pool = attn_pool(h)

        self.ast_label_emb = nn.Embedding(label_vocab, h)
        self.ast_tok_emb = nn.Embedding(token_vocab, h)
        self.treelstm = childsum_treelstm(h, h)
        self.ast_pool = attn_pool(h)

        self.fa_gru = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        self.fa_att = nn.Linear(h, 1)

        self.combine = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn1 = nn.Linear(h, h)
        self.gcn2 = nn.Linear(h, h)

        self.cls = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, 1),
        )

    def encode_ast(self, sp: statement_pack, device: str):
        nodes = sp.ast_nodes
        if len(nodes) == 0:
            return torch.zeros(self.h, device=device)

        children = dict(sp.ast_children)
        for n in nodes:
            children.setdefault(n, [])

        remaining = set(nodes)
        done = set()
        order = []

        while remaining:
            progressed = False
            for n in list(remaining):
                chs = children.get(n, [])
                chs = [c for c in chs if c in nodes]
                if all((c in done) for c in chs):
                    order.append(n)
                    remaining.remove(n)
                    done.add(n)
                    progressed = True
            if not progressed:
                order.extend(list(remaining))
                break

        h_map = {}
        c_map = {}

        for n in order:
            lid = sp.ast_node_label_ids.get(n, 0)
            lbl_e = self.ast_label_emb(torch.tensor(lid, device=device))
            toks = sp.ast_node_tok_ids.get(n, [])
            if len(toks) == 0:
                tok_e = torch.zeros(self.h, device=device)
            else:
                tok_ids = torch.tensor(toks, dtype=torch.long, device=device)
                tok_e = self.ast_tok_emb(tok_ids).mean(dim=0)
            x = 0.5 * lbl_e + 0.5 * tok_e

            chs = [c for c in children.get(n, []) if c in h_map]
            if len(chs) == 0:
                child_h = torch.zeros(0, self.h, device=device)
                child_c = torch.zeros(0, self.h, device=device)
            else:
                child_h = torch.stack([h_map[c] for c in chs], dim=0)
                child_c = torch.stack([c_map[c] for c in chs], dim=0)

            h, c = self.treelstm.node_forward(x, child_h, child_c)
            h_map[n] = h
            c_map[n] = c

        hs = torch.stack(list(h_map.values()), dim=0)
        return self.ast_pool(hs)

    def encode_context(self, stmt_vecs: List[torch.Tensor], sp: statement_pack):
        neis = sp.ctx_neighbors
        if len(neis) == 0:
            return torch.zeros(self.h, device=stmt_vecs[0].device)
        seq = [stmt_vecs[j].detach() for j in neis]
        x = torch.stack(seq, dim=0).unsqueeze(0)
        o, _ = self.seq_ctx(x)
        o = o.squeeze(0)
        return self.ctx_pool(o)

    def feature_attention(self, feats: torch.Tensor):
        x = feats.unsqueeze(0)
        o, _ = self.fa_gru(x)
        o = o.squeeze(0)
        a = self.fa_att(o).squeeze(-1)
        w = torch.softmax(a, dim=0)
        return (w.unsqueeze(-1) * feats).sum(dim=0)

    def gcn_forward(self, x: torch.Tensor, edge_index: List[Tuple[int, int]]):
        n = x.size(0)
        if n <= 1:
            return x

        rows, cols = [], []
        for i, j in edge_index:
            rows += [i, j]
            cols += [j, i]
        rows += list(range(n))
        cols += list(range(n))

        idx = torch.tensor([rows, cols], dtype=torch.long, device=x.device)
        val = torch.ones(len(rows), device=x.device)
        a = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()

        deg = torch.sparse.sum(a, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

        ii = a.indices()[0]
        jj = a.indices()[1]
        norm_val = a.values() * deg_inv_sqrt[ii] * deg_inv_sqrt[jj]
        ahat = torch.sparse_coo_tensor(a.indices(), norm_val, size=a.size()).coalesce()

        h1 = torch.sparse.mm(ahat, self.gcn1(x))
        h1 = f.relu(h1)
        h2 = torch.sparse.mm(ahat, self.gcn2(h1))
        h2 = f.relu(h2)
        return h2

    def forward(self, g: graph_example, device: str):
        stmt_tok_vecs, varn_vecs, vart_vecs, ast_vecs = [], [], [], []
        for sp in g.stmts:
            stmt_tok_vecs.append(self.seq_stmt(sp.stmt_tokens))
            varn_vecs.append(self.seq_varn(sp.varname_tokens))
            vart_vecs.append(self.seq_vart(sp.vartype_tokens))
            ast_vecs.append(self.encode_ast(sp, device=device))

        ctx_vecs = []
        for sp in g.stmts:
            ctx_vecs.append(self.encode_context(stmt_tok_vecs, sp))

        stmt_embs = []
        for i in range(len(g.stmts)):
            feats = torch.stack(
                [stmt_tok_vecs[i], ast_vecs[i], varn_vecs[i], vart_vecs[i], ctx_vecs[i]], dim=0
            )
            stmt_embs.append(self.feature_attention(feats))

        x = torch.stack(stmt_embs, dim=0)

        n = x.size(0)
        nei = [set() for _ in range(n)]
        for a, b in g.edges:
            if a < n and b < n:
                nei[a].add(b)
                nei[b].add(a)

        x2 = []
        for i in range(n):
            if len(nei[i]) == 0:
                nei_mean = torch.zeros(self.h, device=x.device)
            else:
                nei_mean = x[list(nei[i])].mean(dim=0)
            x2.append(self.combine(torch.cat([x[i], nei_mean], dim=-1)))
        x2 = torch.stack(x2, dim=0)

        hg = self.gcn_forward(x2, g.edges)
        g_mean = hg.mean(dim=0)
        g_max = hg.max(dim=0).values
        gvec = torch.cat([g_mean, g_max], dim=-1)
        logits = self.cls(gvec).squeeze(-1)
        return logits


def tune_threshold_mcc(y_true: np.ndarray, p: np.ndarray):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        yhat = (p >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, yhat) if len(np.unique(yhat)) > 1 else 0.0
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_thr = float(thr)
    return best_thr, best_mcc


def metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float):
    yhat = (p >= thr).astype(int)
    acc = accuracy_score(y_true, yhat)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, yhat) if len(np.unique(yhat)) > 1 else 0.0
    try:
        auc = roc_auc_score(y_true, p)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(auc),
        "specificity": float(spec),
        "sensitivity": float(sens),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(thr),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_probs(model: ivd_like, exs: List[graph_example], device: str):
    model.eval()
    ys, ps = [], []
    for g in exs:
        logit = model(g, device=device).detach().float().cpu().item()
        p = 1.0 / (1.0 + math.exp(-logit))
        ys.append(int(g.y))
        ps.append(float(p))
    return np.array(ys, dtype=int), np.array(ps, dtype=float)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[env] device={device} seed={seed}")

    out_dir = str(root / cfg["output"]["dir"])
    ensure_dir(out_dir)

    metrics_json = str(root / cfg["output"]["metrics_json"])
    ckpt_pt = str(root / cfg["output"]["ckpt_pt"])
    ensure_dir(str(Path(metrics_json).parent))
    ensure_dir(str(Path(ckpt_pt).parent))

    with open(str(Path(out_dir) / "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_csv = str(root / cfg["data"]["train_csv"])
    test_csv = str(root / cfg["data"]["test_csv"])
    graph_col = cfg["data"]["graph_col"]
    label_col = cfg["data"]["label_col"]

    h = int(cfg["model"].get("h", 128))
    drop = float(cfg["model"].get("dropout", 0.1))
    token_vocab = int(cfg["model"].get("token_vocab", 50000))
    label_vocab = int(cfg["model"].get("label_vocab", 512))
    max_stmt_tokens = int(cfg["model"].get("max_stmt_tokens", 64))
    max_var_tokens = int(cfg["model"].get("max_var_tokens", 32))
    max_ctx_nei = int(cfg["model"].get("max_ctx_nei", 12))
    max_ast_nodes = int(cfg["model"].get("max_ast_nodes", 64))
    max_ast_depth = int(cfg["model"].get("max_ast_depth", 8))

    epochs = int(cfg["train"].get("epochs", 20))
    lr = float(cfg["train"].get("lr", 2e-3))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    val_frac = float(cfg["train"].get("val_frac", 0.2))

    df_tr = read_csv_any(train_csv)
    df_te = read_csv_any(test_csv)

    if graph_col not in df_tr.columns or label_col not in df_tr.columns:
        raise KeyError(f"train csv must have columns: {graph_col}, {label_col}")
    if graph_col not in df_te.columns or label_col not in df_te.columns:
        raise KeyError(f"test csv must have columns: {graph_col}, {label_col}")

    def build_examples(df: pd.DataFrame):
        exs = []
        dropped = 0
        for _, row in df.iterrows():
            gs = row[graph_col]
            try:
                y = int(row[label_col])
            except Exception:
                dropped += 1
                continue
            if not isinstance(gs, str) or len(gs) < 10:
                dropped += 1
                continue
            ge = build_graph_example(
                gs,
                y,
                token_vocab=token_vocab,
                label_vocab=label_vocab,
                max_stmt_tokens=max_stmt_tokens,
                max_var_tokens=max_var_tokens,
                max_ctx_nei=max_ctx_nei,
                max_ast_nodes=max_ast_nodes,
                max_ast_depth=max_ast_depth,
            )
            if ge is None or len(ge.stmts) == 0:
                dropped += 1
                continue
            exs.append(ge)
        return exs, dropped

    train_all, dropped_tr = build_examples(df_tr)
    test_exs, dropped_te = build_examples(df_te)

    print(f"[graphs] train={len(train_all)} dropped_train={dropped_tr} test={len(test_exs)} dropped_test={dropped_te}")
    if len(train_all) == 0 or len(test_exs) == 0:
        raise RuntimeError("graph build failed for train/test")

    y_all = np.array([g.y for g in train_all], dtype=int)
    idxs = np.arange(len(train_all))
    tr_idx, va_idx = train_test_split(idxs, test_size=val_frac, random_state=seed, stratify=y_all)
    train_exs = [train_all[i] for i in tr_idx]
    val_exs = [train_all[i] for i in va_idx]

    print(f"[split] train={len(train_exs)} val={len(val_exs)} test={len(test_exs)}")

    model = ivd_like(h, token_vocab=token_vocab, label_vocab=label_vocab, dropout=drop).to(device)

    ytr = np.array([g.y for g in train_exs], dtype=int)
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    pos_weight = torch.tensor([(neg / max(pos, 1))], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[class_balance] neg={neg} pos={pos} pos_weight={pos_weight.item():.4f}")

    best_val_mcc = -1e9
    best_state = None
    best_thr = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_exs)
        losses = []

        for g in train_exs:
            opt.zero_grad(set_to_none=True)
            logit = model(g, device=device).view(1)
            y = torch.tensor([float(g.y)], device=device, dtype=torch.float32)
            loss = loss_fn(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        yv, pv = predict_probs(model, val_exs, device=device)
        thr, mccv = tune_threshold_mcc(yv, pv)
        mv = metrics_at_threshold(yv, pv, thr)

        print(
            f"epoch {epoch:02d} loss={np.mean(losses):.4f} "
            f"val_mcc={mccv:.4f} val_f1={mv['f1']:.4f} val_acc={mv['accuracy']:.4f} thr={thr:.2f}"
        )

        if mccv > best_val_mcc:
            best_val_mcc = float(mccv)
            best_thr = float(thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, pt = predict_probs(model, test_exs, device=device)
    mt = metrics_at_threshold(yt, pt, best_thr)

    out = {
        "baseline": "ivdetect_style",
        "seed": int(seed),
        "best_val_mcc": float(best_val_mcc),
        "best_threshold": float(best_thr),
        "test": mt,
        "counts": {
            "train_graphs": int(len(train_all)),
            "val_graphs": int(len(val_exs)),
            "test_graphs": int(len(test_exs)),
            "dropped_train": int(dropped_tr),
            "dropped_test": int(dropped_te),
        },
        "hparams": {
            "h": int(h),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "val_frac": float(val_frac),
            "token_vocab": int(token_vocab),
            "label_vocab": int(label_vocab),
            "max_stmt_tokens": int(max_stmt_tokens),
            "max_var_tokens": int(max_var_tokens),
            "max_ctx_nei": int(max_ctx_nei),
            "max_ast_nodes": int(max_ast_nodes),
            "max_ast_depth": int(max_ast_depth),
        },
    }

    with open(metrics_json, "w", encoding="utf-8") as fobj:
        json.dump(out, fobj, indent=2)
    torch.save(model.state_dict(), ckpt_pt)

    print(f"[paths] metrics_json={metrics_json}")
    print(f"[paths] ckpt_pt={ckpt_pt}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/baselines/ivdetect.yaml")
    args = ap.parse_args()
    main(args.config)

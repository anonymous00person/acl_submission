import json, hashlib
import numpy as np
from collections import Counter
from pathlib import Path

EDGE_LABELS = [
    "AST", "CFG", "CONTAINS", "REACHING_DEF", "EVAL_TYPE",
    "DOMINATE", "POST_DOMINATE", "SOURCE_FILE", "BINDS", "REF",
    "PARAMETER_LINK", "INHERITS_FROM"
]

VERTEX_LABELS = [
    "METHOD", "BLOCK", "FILE", "TYPE", "TYPE_DECL", "MODIFIER",
    "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_RETURN",
    "UNKNOWN", "NAMESPACE", "NAMESPACE_BLOCK", "BINDING", "META_DATA",
    "CALL", "CONTROL_STRUCTURE", "LOCAL", "IDENTIFIER", "LITERAL"
]

FLAG_KEYS = [
    "has_unknown_vertex",
    "has_unparsable_stmt",
    "has_reaching_def",
    "has_cfg",
    "has_ast",
    "file_name_unknown",
    "namespace_is_global",
]

SCALAR_KEYS = [
    "num_vertices",
    "num_edges",
    "avg_edges_per_vertex",
    "ratio_unknown_vertices",
    "num_methods",
    "num_type_decls",
    "num_params_in",
    "num_params_out"
]

FEATURE_KEYS = (
    [f"edgecnt_{k}" for k in EDGE_LABELS] +
    [f"vtxcnt_{k}" for k in VERTEX_LABELS] +
    FLAG_KEYS +
    SCALAR_KEYS
)
FEATURE_DIM = len(FEATURE_KEYS)

def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_graph_features(graphson_obj: dict) -> np.ndarray:
    z = np.zeros(FEATURE_DIM, dtype=np.float32)
    if not isinstance(graphson_obj, dict):
        return z

    try:
        gval = graphson_obj.get("@value", {})
        edges = gval.get("edges", []) or []
        verts = gval.get("vertices", []) or []

        edge_counter = Counter()
        vtx_counter  = Counter()

        has_unknown_vertex = False
        has_unparsable_stmt = False
        has_reaching_def = False
        has_cfg = False
        has_ast = False
        file_name_unknown = False
        namespace_is_global = False

        for e in edges:
            lbl = e.get("label", None)
            if lbl:
                edge_counter[lbl] += 1
                if lbl == "REACHING_DEF": has_reaching_def = True
                if lbl == "CFG": has_cfg = True
                if lbl == "AST": has_ast = True

        unknown_count = 0
        for v in verts:
            lbl = v.get("label", None)
            if lbl:
                vtx_counter[lbl] += 1
                if lbl == "UNKNOWN":
                    has_unknown_vertex = True
                    unknown_count += 1

            props = v.get("properties", {}) or {}

            if lbl == "UNKNOWN":
                pts = props.get("PARSER_TYPE_NAME", {})
                try:
                    pv = pts.get("@value", {}).get("@value", [])
                except Exception:
                    pv = []
                if any(str(x) == "UnparsableStmt" for x in pv):
                    has_unparsable_stmt = True

            if lbl == "FILE":
                name_prop = props.get("NAME", {})
                try:
                    names = name_prop.get("@value", {}).get("@value", [])
                except Exception:
                    names = []
                if any(str(x).strip() in {"<unknown>", ""} for x in names):
                    file_name_unknown = True

            if lbl == "NAMESPACE_BLOCK":
                full_name = props.get("FULL_NAME", {})
                try:
                    vals = full_name.get("@value", {}).get("@value", [])
                except Exception:
                    vals = []
                if any(str(x).strip() == "<global>" or str(x).strip().endswith("<global>") for x in vals):
                    namespace_is_global = True

        num_edges = len(edges)
        num_vertices = len(verts)
        avg_edges_per_vertex = float(num_edges) / max(1, num_vertices)
        ratio_unknown_vertices = float(unknown_count) / max(1, num_vertices)

        feats = {}
        for k in EDGE_LABELS:
            feats[f"edgecnt_{k}"] = float(edge_counter.get(k, 0))
        for k in VERTEX_LABELS:
            feats[f"vtxcnt_{k}"] = float(vtx_counter.get(k, 0))

        feats["has_unknown_vertex"] = 1.0 if has_unknown_vertex else 0.0
        feats["has_unparsable_stmt"] = 1.0 if has_unparsable_stmt else 0.0
        feats["has_reaching_def"] = 1.0 if has_reaching_def else 0.0
        feats["has_cfg"] = 1.0 if has_cfg else 0.0
        feats["has_ast"] = 1.0 if has_ast else 0.0
        feats["file_name_unknown"] = 1.0 if file_name_unknown else 0.0
        feats["namespace_is_global"] = 1.0 if namespace_is_global else 0.0

        feats["num_vertices"] = float(num_vertices)
        feats["num_edges"] = float(num_edges)
        feats["avg_edges_per_vertex"] = float(avg_edges_per_vertex)
        feats["ratio_unknown_vertices"] = float(ratio_unknown_vertices)
        feats["num_methods"] = float(vtx_counter.get("METHOD", 0))
        feats["num_type_decls"] = float(vtx_counter.get("TYPE_DECL", 0))
        feats["num_params_in"] = float(vtx_counter.get("METHOD_PARAMETER_IN", 0))
        feats["num_params_out"] = float(vtx_counter.get("METHOD_PARAMETER_OUT", 0))

        return np.array([feats.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    except Exception:
        return z

def features_from_graphson_str(graphson_str: str, cache_dir: str) -> np.ndarray:
    if graphson_str is None or str(graphson_str).strip() == "":
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    key = hashlib.md5(graphson_str.encode("utf-8", errors="ignore")).hexdigest()
    cache_file = cache_dir_p / f"{key}.npy"

    if cache_file.exists():
        try:
            return np.load(str(cache_file))
        except Exception:
            pass

    graph = _safe_json_load(graphson_str)
    vec = extract_graph_features(graph)
    try:
        np.save(str(cache_file), vec)
    except Exception:
        pass
    return vec

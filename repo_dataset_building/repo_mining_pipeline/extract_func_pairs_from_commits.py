import os
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from repo_config import BASE_DIR, HUNKS_CSV, FUNC_PAIRS_CSV, JAVAPARSER_DIR, EXTRACTOR_JAR, EXTRACTOR_CLASS
from utils import clone_or_open, get_file_at_commit, unique_keep_order


def extract_func_at_line(source: str, line_no: int, tmp_path: Path) -> str:
    if not isinstance(source, str) or not source.strip() or not isinstance(line_no, int) or line_no <= 0:
        return ""

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(source, encoding="utf-8", errors="ignore")

    cp = f".:{EXTRACTOR_JAR}"
    cmd = ["java", "-cp", cp, EXTRACTOR_CLASS, str(tmp_path), str(line_no)]

    try:
        p = subprocess.run(
            cmd,
            cwd=str(JAVAPARSER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except Exception:
        return ""

    if p.returncode != 0:
        return ""

    return (p.stdout or "").strip()


def main() -> None:
    hunks = pd.read_csv(HUNKS_CSV)

    required = {"repo", "sha", "old_start", "new_start", "patch"}
    miss = required - set(hunks.columns)
    if miss:
        raise KeyError(f"missing columns in HUNKS_CSV: {sorted(miss)}")

    out_rows = []

    for repo_slug, g in hunks.groupby("repo"):
        repo = clone_or_open(repo_slug, BASE_DIR)

        for _, r in tqdm(g.iterrows(), total=len(g), desc=f"func_pairs {repo_slug}"):
            sha = str(r["sha"]).strip()
            if not sha:
                continue

            try:
                commit = repo.commit(sha)
            except Exception:
                try:
                    repo.git.fetch("origin", sha)
                    commit = repo.commit(sha)
                except Exception:
                    continue

            if not commit.parents:
                continue

            parent = commit.parents[0]
            old_start = int(r["old_start"]) if pd.notna(r["old_start"]) else None
            new_start = int(r["new_start"]) if pd.notna(r["new_start"]) else None
            if not old_start or not new_start:
                continue

            try:
                diff_index = commit.diff(parent, create_patch=True)
            except Exception:
                continue

            tmp_old = Path(JAVAPARSER_DIR) / "tmp_old.java"
            tmp_new = Path(JAVAPARSER_DIR) / "tmp_new.java"

            for d in diff_index:
                a_path = d.a_path or ""
                b_path = d.b_path or ""
                java_path = a_path if a_path.endswith(".java") else b_path if b_path.endswith(".java") else ""
                if not java_path:
                    continue

                old_code = get_file_at_commit(repo, parent.hexsha, java_path)
                new_code = get_file_at_commit(repo, commit.hexsha, java_path)

                old_func = extract_func_at_line(old_code, old_start, tmp_old) if old_code else ""
                new_func = extract_func_at_line(new_code, new_start, tmp_new) if new_code else ""

                if not old_func and not new_func:
                    continue

                out_rows.append(
                    {
                        "repo": repo_slug,
                        "sha": sha,
                        "file": java_path,
                        "old_start": old_start,
                        "new_start": new_start,
                        "old_func": old_func,
                        "new_func": new_func,
                    }
                )

    df_out = pd.DataFrame(out_rows)
    if len(df_out):
        df_out = df_out.drop_duplicates(subset=["repo", "sha", "file", "old_start", "new_start"], keep="first")

    df_out.to_csv(FUNC_PAIRS_CSV, index=False)
    print(f"saved: {FUNC_PAIRS_CSV} | rows={len(df_out)}")


if __name__ == "__main__":
    main()

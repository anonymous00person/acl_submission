# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from git import Repo

from ghsa_config import get_build_config, get_paths
from git_utils import cleanup_repo_dir, clone_or_open_repo, ensure_commit
from javaparser_runner import pick_start_lines_from_patch, run_extractor


def extract_pairs_for_commit(
    repo: Repo,
    sha: str,
    javaparser_dir: Path,
    javaparser_jar: str,
    extractor_class: str,
    tmp_old: str,
    tmp_new: str,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not ensure_commit(repo, sha):
        return out

    commit = repo.commit(sha)
    if not commit.parents:
        return out
    parent = commit.parents[0]

    try:
        diffs = commit.diff(parent, create_patch=True)
    except Exception:
        return out

    for d in diffs:
        java_path = d.a_path or d.b_path
        if not java_path or not java_path.endswith(".java"):
            continue

        patch_bytes = d.diff
        if not patch_bytes:
            continue
        try:
            patch_text = patch_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue

        old_code: Optional[str] = None
        new_code: Optional[str] = None

        try:
            repo.git.ls_tree(parent.hexsha, java_path)
            old_code = repo.git.show(f"{parent.hexsha}:{java_path}")
        except Exception:
            old_code = None

        try:
            repo.git.ls_tree(commit.hexsha, java_path)
            new_code = repo.git.show(f"{commit.hexsha}:{java_path}")
        except Exception:
            new_code = None

        if old_code is None and new_code is None:
            continue

        for old_start, new_start in pick_start_lines_from_patch(patch_text):
            old_func = ""
            new_func = ""

            if old_code is not None:
                old_func = run_extractor(
                    javaparser_dir=javaparser_dir,
                    javaparser_jar=javaparser_jar,
                    extractor_class=extractor_class,
                    source_code=old_code,
                    line_no=old_start,
                    tmp_filename=tmp_old,
                    timeout_sec=timeout_sec,
                )

            if new_code is not None:
                new_func = run_extractor(
                    javaparser_dir=javaparser_dir,
                    javaparser_jar=javaparser_jar,
                    extractor_class=extractor_class,
                    source_code=new_code,
                    line_no=new_start,
                    tmp_filename=tmp_new,
                    timeout_sec=timeout_sec,
                )

            if old_func or new_func:
                out.append(
                    {
                        "file": java_path,
                        "old_start": int(old_start),
                        "new_start": int(new_start),
                        "old_func": old_func,
                        "new_func": new_func,
                    }
                )

    return out


def main() -> None:
    paths = get_paths()
    cfg = get_build_config()

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)

    if not paths.ghsa_csv.exists():
        pd.DataFrame().to_csv(paths.ghsa_csv, index=False, encoding="utf-8")
        return

    meta = pd.read_csv(paths.ghsa_csv)
    if meta.empty:
        pd.DataFrame().to_csv(paths.pairs_csv, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(paths.flat_csv, index=False, encoding="utf-8")
        return

    for col in ["repo_url", "fix_commit"]:
        if col not in meta.columns:
            raise KeyError(col)

    meta = meta.dropna(subset=["repo_url", "fix_commit"]).reset_index(drop=True)
    if meta.empty:
        pd.DataFrame().to_csv(paths.pairs_csv, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(paths.flat_csv, index=False, encoding="utf-8")
        return

    all_pairs: List[Dict[str, Any]] = []

    for i, (repo_url, group) in enumerate(meta.groupby("repo_url"), start=1):
        if cfg.max_repos is not None and i > cfg.max_repos:
            break

        repo, repo_path = clone_or_open_repo(str(repo_url), paths.repo_dir)
        if repo is None or repo_path is None:
            continue

        try:
            group_u = group.drop_duplicates(subset=["fix_commit"])
            if cfg.max_commits_per_repo is not None:
                group_u = group_u.head(cfg.max_commits_per_repo)

            for _, row in group_u.iterrows():
                sha = str(row["fix_commit"]).strip()
                if not sha:
                    continue

                pairs = extract_pairs_for_commit(
                    repo=repo,
                    sha=sha,
                    javaparser_dir=paths.tools_dir,
                    javaparser_jar=cfg.javaparser_jar,
                    extractor_class=cfg.extractor_class,
                    tmp_old=cfg.tmp_old,
                    tmp_new=cfg.tmp_new,
                    timeout_sec=cfg.request_timeout_sec,
                )

                if not pairs:
                    continue

                for p in pairs:
                    p["ghsa_id"] = row.get("ghsa_id")
                    p["cve_id"] = row.get("cve_id")
                    p["severity"] = row.get("severity")
                    p["cwes"] = row.get("cwes")
                    p["package"] = row.get("package")
                    p["repo_url"] = repo_url
                    p["fix_commit"] = sha
                    p["published_at"] = row.get("published_at")
                    all_pairs.append(p)

        finally:
            try:
                repo.close()
            except Exception:
                pass
            if cfg.cleanup_repos:
                cleanup_repo_dir(repo_path)

    df_pairs = pd.DataFrame(all_pairs)
    df_pairs.to_csv(paths.pairs_csv, index=False, encoding="utf-8")

    flat_rows: List[Dict[str, Any]] = []
    for _, r in df_pairs.iterrows():
        meta_row = {
            "ghsa_id": r.get("ghsa_id"),
            "cve_id": r.get("cve_id"),
            "severity": r.get("severity"),
            "cwes": r.get("cwes"),
            "package": r.get("package"),
            "repo_url": r.get("repo_url"),
            "commit": r.get("fix_commit"),
            "file": r.get("file"),
            "published_at": r.get("published_at"),
        }

        old_code = r.get("old_func")
        if isinstance(old_code, str) and old_code.strip():
            flat_rows.append(
                {
                    **meta_row,
                    "line": r.get("old_start"),
                    "code": old_code,
                    "version": "vulnerable",
                    "is_vuln": 1,
                }
            )

        new_code = r.get("new_func")
        if isinstance(new_code, str) and new_code.strip():
            flat_rows.append(
                {
                    **meta_row,
                    "line": r.get("new_start"),
                    "code": new_code,
                    "version": "patched",
                    "is_vuln": 0,
                }
            )

    pd.DataFrame(flat_rows).to_csv(paths.flat_csv, index=False, encoding="utf-8")







if __name__ == "__main__":
    main()

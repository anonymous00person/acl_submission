import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Dict, List, Optional, Tuple

from git import Repo
from git.exc import GitCommandError

def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def repo_slug_to_dir(slug: str) -> str:
    return slug.replace("/", "_")

def clone_or_open(slug: str, base_dir: str) -> Repo:
    ensure_dir(base_dir)
    repo_dir = Path(base_dir) / repo_slug_to_dir(slug)
    url = f"https://github.com/{slug}.git"
    if not (repo_dir / ".git").exists():
        Repo.clone_from(url, str(repo_dir))
    return Repo(str(repo_dir))

def compile_keyword_regex(keywords: List[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in keywords]
    patt = r"|".join([rf"\b{e}\b" for e in escaped if e])
    return re.compile(patt, re.IGNORECASE)

def is_security_message(msg: str, rx: re.Pattern) -> bool:
    if not isinstance(msg, str):
        return False
    return rx.search(msg) is not None

def iter_commits(repo: Repo):
    for c in repo.iter_commits():
        yield c

def get_commit_patch(repo: Repo, commit_sha: str) -> str:
    c = repo.commit(commit_sha)
    parent = c.parents[0] if c.parents else None
    if parent is None:
        return ""
    diffs = c.diff(parent, create_patch=True)
    parts = []
    for d in diffs:
        try:
            parts.append(d.diff.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(parts)

def parse_hunk_headers(patch_text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not isinstance(patch_text, str):
        return out
    for line in patch_text.splitlines():
        if not line.startswith("@@"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            old_part = parts[1]
            new_part = parts[2]
            old_start = int(old_part.split(",")[0].lstrip("-"))
            new_start = int(new_part.split(",")[0].lstrip("+"))
            out.append((old_start, new_start))
        except Exception:
            continue
    return out

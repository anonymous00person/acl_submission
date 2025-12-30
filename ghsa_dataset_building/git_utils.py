# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import Optional, Tuple

from git import Repo
from git.exc import GitCommandError


def safe_repo_name(repo_url: str) -> Optional[str]:
    if not isinstance(repo_url, str) or not repo_url.startswith("http"):
        return None
    name = repo_url.rstrip("/").split("/")[-1].strip()
    return name or None


def clone_or_open_repo(repo_url: str, repo_dir: Path) -> Tuple[Optional[Repo], Optional[Path]]:
    repo_dir.mkdir(parents=True, exist_ok=True)
    name = safe_repo_name(repo_url)
    if not name:
        return None, None

    path = repo_dir / name
    if not (path / ".git").exists():
        try:
            repo = Repo.clone_from(repo_url, str(path), depth=1)
            return repo, path
        except GitCommandError:
            return None, None
        except Exception:
            return None, None

    try:
        return Repo(str(path)), path
    except Exception:
        return None, None


def ensure_commit(repo: Repo, sha: str) -> bool:
    try:
        repo.commit(sha)
        return True
    except Exception:
        try:
            repo.git.fetch("origin", sha)
            repo.commit(sha)
            return True
        except Exception:
            return False


def cleanup_repo_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

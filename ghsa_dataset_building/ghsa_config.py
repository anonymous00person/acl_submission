# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    out_dir: Path
    repo_dir: Path
    tools_dir: Path
    ghsa_csv: Path
    pairs_csv: Path
    flat_csv: Path


@dataclass(frozen=True)
class GHSAConfig:
    base_url: str
    headers: Dict[str, str]
    ecosystem: str
    per_page: int
    sort: str
    direction: str
    advisory_type: str


@dataclass(frozen=True)
class BuildConfig:
    max_repos: Optional[int]
    max_commits_per_repo: Optional[int]
    cleanup_repos: bool
    sleep_sec: float
    request_timeout_sec: int
    javaparser_jar: str
    extractor_class: str
    tmp_old: str
    tmp_new: str


def get_paths() -> Paths:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    out_dir = root / "outputs"
    repo_dir = root / "repos"
    tools_dir = root / "tools"
    return Paths(
        root=root,
        data_dir=data_dir,
        out_dir=out_dir,
        repo_dir=repo_dir,
        tools_dir=tools_dir,
        ghsa_csv=data_dir / "ghsa_maven_advisories.csv",
        pairs_csv=out_dir / "ghsa_java_func_pairs.csv",
        flat_csv=out_dir / "ghsa_java_funcs_flat.csv",
    )


def get_ghsa_config() -> GHSAConfig:
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return GHSAConfig(
        base_url="https://api.github.com/advisories",
        headers=headers,
        ecosystem="maven",
        per_page=100,
        sort="published",
        direction="desc",
        advisory_type="reviewed",
    )


def get_build_config() -> BuildConfig:
    return BuildConfig(
        max_repos=60,
        max_commits_per_repo=None,
        cleanup_repos=True,
        sleep_sec=0.5,
        request_timeout_sec=60,
        javaparser_jar="javaparser-core-3.25.4.jar",
        extractor_class="FunctionExtractor",
        tmp_old="tmp_old.java",
        tmp_new="tmp_new.java",
    )

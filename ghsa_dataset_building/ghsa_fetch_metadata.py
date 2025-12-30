# -*- coding: utf-8 -*-

import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ghsa_config import get_build_config, get_ghsa_config, get_paths


def _repo_from_github_url(url: str) -> Optional[str]:
    if not isinstance(url, str) or "github.com" not in url:
        return None
    m = re.search(r"https://github\.com/([^/]+/[^/#\.]+)", url)
    if not m:
        return None
    return f"https://github.com/{m.group(1)}"


def extract_repo_from_advisory(advisory: Dict[str, Any]) -> Optional[str]:
    src = advisory.get("source_code_location")
    if isinstance(src, str):
        r = _repo_from_github_url(src)
        if r:
            return r

    for url in (advisory.get("references") or []):
        r = _repo_from_github_url(url)
        if r:
            return r

    return None


def extract_commit_from_refs(refs: str) -> Optional[str]:
    if not isinstance(refs, str):
        return None

    for url in refs.split(";"):
        url = url.strip()
        m = re.search(r"github\.com/[^/]+/[^/]+/commit/([0-9a-f]{7,40})", url)
        if m:
            return m.group(1)

    return None


def fetch_maven_advisories(
    max_rows: int,
    sleep_sec: float,
    timeout_sec: int,
) -> pd.DataFrame:
    cfg = get_ghsa_config()
    rows: List[Dict[str, Any]] = []
    after: Optional[str] = None

    base_params: Dict[str, Any] = {
        "ecosystem": "maven",
        "per_page": cfg.per_page,
        "sort": "published",
        "direction": "desc",
        "type": "reviewed",
    }

    while len(rows) < max_rows:
        params = dict(base_params)
        if after:
            params["after"] = after

        r = requests.get(cfg.base_url, headers=cfg.headers, params=params, timeout=timeout_sec)
        if r.status_code != 200:
            break

        advisories = r.json()
        if not advisories:
            break

        link_header = r.headers.get("Link", "")
        next_after: Optional[str] = None
        if 'rel="next"' in link_header:
            m = re.search(r"after=([^&>]+)", link_header)
            if m:
                next_after = m.group(1)

        for adv in advisories:
            ghsa_id = adv.get("ghsa_id")
            cve_id = adv.get("cve_id")
            summary = adv.get("summary")
            description = adv.get("description")
            severity = adv.get("severity")
            html_url = adv.get("html_url")
            published_at = adv.get("published_at")

            cwe_ids = [c.get("cwe_id") for c in (adv.get("cwes") or []) if c.get("cwe_id")]
            cwes = ",".join(cwe_ids)

            repo_url = extract_repo_from_advisory(adv)
            refs_list = adv.get("references") or []
            refs = ";".join(refs_list)

            vulns = adv.get("vulnerabilities") or []
            if not vulns:
                continue

            for v in vulns:
                pkg = v.get("package") or {}
                if pkg.get("ecosystem") != "maven":
                    continue

                rows.append(
                    {
                        "ghsa_id": ghsa_id,
                        "cve_id": cve_id,
                        "ecosystem": "maven",
                        "package": pkg.get("name"),
                        "severity": severity,
                        "cwes": cwes,
                        "summary": summary,
                        "description": description,
                        "html_url": html_url,
                        "repo_url": repo_url,
                        "vulnerable_version_range": v.get("vulnerable_version_range"),
                        "first_patched_version": v.get("first_patched_version"),
                        "references": refs,
                        "published_at": published_at,
                    }
                )

                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

        if not next_after:
            break

        after = next_after
        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["fix_commit"] = df["references"].apply(extract_commit_from_refs)
    return df


def main() -> None:
    paths = get_paths()
    bcfg = get_build_config()

    paths.data_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_maven_advisories(
        max_rows=5000,
        sleep_sec=bcfg.sleep_sec,
        timeout_sec=bcfg.request_timeout_sec,
    )

    if df.empty:
        df.to_csv(paths.ghsa_csv, index=False, encoding="utf-8")
        return

    df = df[df["published_at"].notna()]
    df = df[df["published_at"] >= "2020-01-01"]
    df = df[df["severity"].isin(["high", "critical"])]

    df = df[df["repo_url"].notna() & df["fix_commit"].notna()].reset_index(drop=True)

    df.to_csv(paths.ghsa_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()

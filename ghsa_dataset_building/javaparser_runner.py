# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
from typing import Optional


def run_extractor(
    javaparser_dir: Path,
    javaparser_jar: str,
    extractor_class: str,
    source_code: str,
    line_no: int,
    tmp_filename: str,
    timeout_sec: int,
) -> str:
    tmp_path = javaparser_dir / tmp_filename
    try:
        tmp_path.write_text(source_code, encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    cp = f".:{javaparser_jar}"
    cmd = ["java", "-cp", cp, extractor_class, str(tmp_path), str(int(line_no))]

    try:
        r = subprocess.run(
            cmd,
            cwd=str(javaparser_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except Exception:
        return ""

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    if r.returncode != 0:
        return ""

    return (r.stdout or "").strip()


def pick_start_lines_from_patch(patch_text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for ln in (patch_text or "").splitlines():
        if not ln.startswith("@@"):
            continue
        try:
            parts = ln.split()
            old_part = parts[1]
            new_part = parts[2]
            old_start = int(old_part.split(",")[0].lstrip("-"))
            new_start = int(new_part.split(",")[0].lstrip("+"))
            pairs.append((old_start, new_start))
        except Exception:
            continue
    return pairs

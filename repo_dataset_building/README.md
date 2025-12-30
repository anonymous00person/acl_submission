# Repo-mined Java Vulnerability Dataset 

This folder contains a lightweight outline of how we built **Dataset 1 (Repo-mined)** used in our paper.  

---

## What this dataset is

We build a **function-level Java dataset** from real-world GitHub repositories by:
1) mining candidate security-fix commits,
2) extracting **before/after** function pairs around changed hunks using JavaParser, and
3) assigning labels via **heuristic CWE pattern matching** (applied to the vulnerable side only).

The dataset supports:
- function-level vulnerability classification
- patch-aware analysis via (old_func, new_func) pairs
- labeling via documented heuristics + CodeQL policy

---

## Main contribution (Dataset 1)

### 1) Commit mining from GitHub repos
- Start from a fixed list of Java repositories.
- Scan commit messages with security keywords (e.g., `cve`, `security`, `patch`, `xss`, `rce`, `auth`).
- Output: commit table containing `(repo, sha, date, message)`.

### 2) Patch hunk extraction
- For each mined commit, compute `git diff` against its parent commit.
- Keep only `.java` files.
- Parse each hunk header (`@@ -old_start,+new_start @@`) to obtain line anchors.
- Output: file+hunk rows `(sha, file, old_start, new_start, patch_hunk)`.

### 3) Function-level extraction (JavaParser line-to-function)
We map each hunk’s start line to the enclosing method/class block using a JavaParser helper:
- `old_func`: extracted from the **parent** snapshot at `old_start`
- `new_func`: extracted from the **fix** commit snapshot at `new_start`

This yields a pair-level table with:
`(commit, file, old_start->new_start, old_func, new_func)`.

> Note: In our labeling, we use **only `old_func`** as the vulnerable-side text.

### 4) Heuristic CWE labeling (used in this dataset)
We apply heuristic CWE patterns to **old_func only**:
- `old_func` → heuristic CWE tags (one or multiple)


### 5) Deduplication
Before training/testing, we optionally deduplicate by exact function text:
- drop duplicates using the `code`/`old_func` field.

---

## Folder contents (what to look at)

### Dataset building outline / scripts (high-level)
- `repo_mining_pipeline/`
  - `mine_security_commits.py` — keyword-based commit mining (cleaned)
  - `extract_func_pairs_from_commits.py` — diff hunk parsing + function-pair extraction (outline)

### JavaParser helper
- `tools/FunctionExtractor.java` — extracts the enclosing method/class for a given line number

### Labeling (heuristic-focused)
- `labeling/`
  - `heuristic_patterns.py` — **CWE_PATTERNS only**
  - `apply_heuristic_labels.py` — applies patterns to **old_func** only
  - `codeql/` — 
    - `README.md` — CodeQL CLI + pack version 
   

### Dedup
-  deduplicate by exact function text (minimal)




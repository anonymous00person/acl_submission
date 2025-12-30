# ACL 2026 Anonymous Artifacts: GHSA-based Java Vulnerability Detection

This repository contains the code and artifacts to reproduce our main evaluation and baseline comparisons on a GHSA-based Java vulnerability dataset.

---

## Repository Structure

- `data/` : Dataset files used in our experiments (see **Dataset Availability** below).
- `configs/` : YAML configuration files for evaluation and baselines.
- `scripts/` : One-line runnable scripts for evaluation and baselines.
- `artifacts/` : Model/adapter artifacts (if included).
- `results/` (optional) : Saved outputs, predictions, and summary tables (if included).

---

## Dataset Availability

The dataset used in this work is provided under:

- `data/`

> Note: If you do not see the dataset files in `data/` after cloning, they may have been excluded from version control for size/privacy reasons. In that case, please follow the instructions in the paper/appendix for obtaining the dataset (or contact the authors via the anonymized link).

---

## Environment Setup

We recommend using a clean virtual environment.

### 1) Install PyTorch (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

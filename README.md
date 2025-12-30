# Overview of our project 


---

## Repository Structure

- `data/` : Dataset files used in our experiments.
- `configs/` : YAML configuration files for evaluation.
- `scripts/` : training and testing files
- `artifacts/` : Model/adapter artifacts (best model trained saved).
- `repo_dataset_building/` : build of Dataset1.
- `ghsa_dataset_building/` : build of Dataset2.

---

## Dataset Availability

All the datasets used in this work is provided under:

- `data/`: files are zipped for larger sizes

---

## Environment Setup

We recommend using a clean virtual environment.

### 1) Install PyTorch (CUDA 11.8)
```bash
'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
'pip install -r requirements.txt'



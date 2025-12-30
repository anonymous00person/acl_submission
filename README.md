# Overview of our project 


---

## Repository Structure

- `data/` : Dataset files used in our experiments.
- `configs/` : YAML configuration files for training and testing.
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

'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
'pip install -r requirements.txt

### 2) Run the following command for execution (Qwen Coder2.5- 1.5B model- performed best)
'python scripts/eval.py --config configs/eval.yaml'

### 3) Aditionally for training (for our dataset already saved trained model under artifacts folder)
'python scripts/train.py --config configs/train.yaml'


---

## Notes
Our experiments are conducted in the same environment and setup . Detailed experimental results are included in our paper. 


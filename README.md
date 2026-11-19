# Wine Classification — Six-student MLOps practical

This project implements the examination aim: Wine Classification with the
Scikit-learn Wine Dataset, DVC data versioning, automated data pipelines,
a Feast feature store, MLflow experiment tracking/model selection, and GitHub.

## Architecture

1. Student 1: download/export the sklearn Wine dataset and version it with DVC.
2. Student 2: validate, split, and prepare features.
3. Student 3: register and query features through a local Feast feature store.
4. Student 4: implement candidate classifiers and evaluation.
5. Student 5: run MLflow experiments, compare models, and save the best model.
6. Student 6: connect the steps with DVC and document final GitHub integration.

## Environment

Recommended: Python 3.10 or 3.11. Run commands from the repository root.

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Feast or another dependency fails to install on the lab computer, check the
Python version and internet access first. Do not silently skip the feature-store
step in the examination.

## First-time GitHub setup

One coordinator creates an empty GitHub repository, clones it, adds the starter
files (`requirements.txt`, `.gitignore`, `README.md`, and empty `src/`,
`data/`, `models/` folders if desired), then pushes `main`.

```bash
git clone https://github.com/USERNAME/wine-mlops-practical.git
cd wine-mlops-practical
git switch -c student-1
```

Each student should clone the SAME repository and create their own branch from
the initial `main` commit. Each student adds only the files assigned below.
Commit and push their branch. The coordinator merges the six branches into main.

## Student ownership / files

| Student | Branch | Files owned |
|---|---|---|
| 1 | `student-1` | `src/load_data.py`, DVC initialization and raw-data tracking |
| 2 | `student-2` | `src/preprocess.py` |
| 3 | `student-3` | `src/feature_store.py`, `feature_repo/feature_store.yaml` |
| 4 | `student-4` | `src/model_training.py` |
| 5 | `student-5` | `src/mlflow_tracking.py` |
| 6 | `student-6` | `dvc.yaml`, `params.yaml`, final integration/testing |

Create these folders on main before students start:
```bash
mkdir -p src data/raw data/processed models feature_repo
```
On Windows, create the same folders in File Explorer if `mkdir -p` is unavailable.

## Student 1 — dataset and DVC

Create `src/load_data.py` with the code in the file. It exports the built-in
Scikit-learn Wine dataset to `data/raw/wine.csv`.

```bash
git switch -c student-1
python src/load_data.py
dvc init
git add src/load_data.py .dvc .gitignore
git commit -m "Add Wine dataset loading and initialize DVC"
git push -u origin student-1
```

If `dvc init` says DVC is already initialized, do not run it again. The raw
CSV is declared as an output in Student 6's `dvc.yaml`; DVC will version it
through the pipeline. For a shared remote, configure one if the examiner/lab provides it:
```bash
dvc remote add -d storage <REMOTE-URL>
dvc push
```
Do not invent a remote URL. If no remote is provided, commit the `.dvc` pointer
and explain that the cache is local to the machine.

## Student 2 — preprocessing

Create `src/preprocess.py`. It reads the raw CSV, checks columns/missing values,
creates a reproducible stratified train/test split, and writes:
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/wine_features.parquet`

Run:
```bash
git switch -c student-2
python src/preprocess.py
git add src/preprocess.py
git commit -m "Add wine data preprocessing"
git push -u origin student-2
```
Generated processed files are reproducible and are ignored by `.gitignore`;
DVC will track them through the pipeline in Student 6's work.

## Student 3 — Feast feature store

Create `src/feature_store.py` and `feature_repo/feature_store.yaml`.
The feature view is defined in Python in `src/feature_store.py`, and the local
Feast registry/online store are created under `feature_repo/data/`.

Run after Student 2's preprocessing:
```bash
git switch -c student-3
python src/feature_store.py apply
python src/feature_store.py materialize
python src/feature_store.py check
git add src/feature_store.py feature_repo/feature_store.yaml
git commit -m "Add Feast wine feature store"
git push -u origin student-3
```

## Student 4 — model training

Create `src/model_training.py`. It loads training/testing rows by asking Feast
for historical features, then trains Logistic Regression, Random Forest, and
Support Vector Machine models. It returns their metrics and fitted estimators.

Run:
```bash
git switch -c student-4
python src/model_training.py
git add src/model_training.py
git commit -m "Add wine classification model training"
git push -u origin student-4
```

## Student 5 — MLflow tracking and model selection

Create `src/mlflow_tracking.py`. It logs each model's parameters and metrics,
compares test accuracy, selects the best candidate, logs it to MLflow, and
exports the selected estimator to `models/best_model.joblib`.

Run:
```bash
git switch -c student-5
python src/mlflow_tracking.py
git add src/mlflow_tracking.py
git commit -m "Add MLflow tracking and best-model selection"
git push -u origin student-5
```

View the experiment:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open the local URL printed by MLflow (usually http://127.0.0.1:5000).

## Student 6 — DVC pipeline and integration

Create `params.yaml` and `dvc.yaml` on the `student-6` branch. Since the
other students' files are not yet present on this branch, do not try to run the
complete pipeline here. Commit and push the integration files:
```bash
git switch -c student-6
git add dvc.yaml params.yaml
git commit -m "Add reproducible DVC MLOps pipeline"
git push -u origin student-6
```
The coordinator runs `dvc repro` on `main` after merging all six branches.

The DVC pipeline has four stages: dataset export, preprocessing, Feast setup,
and MLflow model training/selection.

## Merge all branches (coordinator)

After students push:
```bash
git switch main
git pull origin main
git merge --no-ff origin/student-1
git merge --no-ff origin/student-2
git merge --no-ff origin/student-3
git merge --no-ff origin/student-4
git merge --no-ff origin/student-5
git merge --no-ff origin/student-6
git push origin main
```
If remote branches are not local yet, run `git fetch origin` first. Resolve any
conflict, test, then commit the merge resolution.

## Final execution after merge

```bash
pip install -r requirements.txt
dvc repro
mlflow ui --backend-store-uri ./mlruns
```
Expected outputs:
- raw Wine CSV tracked by DVC
- processed train/test CSV and Feast Parquet source
- Feast feature view registered and feature retrieval working
- MLflow experiment with three candidate runs
- selected model at `models/best_model.joblib`

## Viva points

- DVC versions datasets and pipeline outputs; Git versions source code and DVC
  pointer files.
- Feast is the feature store: it defines, registers, and serves reusable features.
- MLflow records experiment parameters, metrics, and model artifacts.
- A DVC pipeline makes the workflow reproducible and reruns stages whose
  dependencies changed.
- Git branches let six contributors work independently; merging combines their
  commits into the final `main` branch.

## Important

The six branches must add distinct files as assigned. Do not let multiple
students edit the same file. Because later sections depend on earlier sections,
students can write their assigned files independently, but full execution/testing
happens after the coordinator merges all six branches into `main`.

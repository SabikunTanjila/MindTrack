# MindTrack — Google Colab edition

A project for lifestyle clustering and prediction of four dataset stress
labels: **Low, Medium, High, Very High**.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload **MindTrack_Colab.ipynb** using **File → Upload notebook**.
3. Connect to a hosted Python 3 CPU runtime and run cells in order.
4. Upload **data/raw/mental_health_dataset.csv** when asked.
5. Complete training/evaluation and open the dashboard inside the notebook.
6. Download **MindTrack_results.zip** before disconnecting.

The notebook embeds the application source. No local Python setup, local server,
Node, GPU, API key, Drive mount, or project ZIP upload is needed. Your local
folder holds editable files; execution takes place in Colab.

See [the step-by-step Colab guide](docs/colab-guide.md).

## Implemented workflow

- Audit, negative-value cleaning, duplicate/profile leakage prevention.
- Stratified 60/20/20 split and training-only preprocessing.
- K-Means selection with silhouette/inertia, profile interpretation, and PCA.
- KNN, Logistic Regression, and Random Forest with stratified CV tuning.
- Validation macro F1 selection followed by held-out test evaluation.
- Majority baseline, confusion matrices, multiclass AUC, and other metrics.
- Global permutation importance and input/median comparisons.
- FastAPI health, model-info, metrics, and prediction endpoints.
- Responsive assessment/results/research dashboard served inside Colab.
- Behavior tests and downloadable fitted models, metrics, plots, and versions.
- A generated lab report and presentation notes using the run's actual results.

## Project map

```text
MindTrack_Colab.ipynb          Complete self-contained Colab workflow
data/raw/                    User-provided CSV
data/processed/              Cleaned data generated in Colab
notebooks/                   Four independent topic notebooks
ml/config.py                 Dataset features, labels, bounds
ml/preprocessing.py          Cleaning, pipelines, stratified splits
ml/clustering.py             K-Means selection and PCA
ml/train.py                  Tuning and validation selection
ml/evaluate.py               Holdout metrics and global importance
ml/reporting.py              Plots, metadata, artifact export
ml/writeup.py                Report and presentation notes from measured results
backend/                     Schemas, inference, suggestions, FastAPI
frontend/                    HTML/CSS/JavaScript dashboard
tests/                       Data and integrated model/API tests
models/                      Fitted artifacts generated in Colab
reports/                     Generated figures and metrics
docs/                        Dataset notes, design, plan, Colab guide
scripts/build-notebooks.ps1  Rebuild notebook files after source edits
requirements.txt             Colab dependencies
```

## Dataset and interpretation

The CSV has 5,000 records, 13 columns, 2 exact duplicates, and 10 negative
physical-activity values. Ages range from 18 to 24. Source, license, collection
method, and synthetic status are unverified. Study/activity time windows also
require source confirmation.

Mental_Health_Score and the target are excluded from model inputs. Gender and
country are omitted. K-Means groups are lifestyle profiles, not stress classes.
Model probabilities are uncalibrated. This is an educational estimate of dataset
labels, not a clinical assessment. See [dataset notes](docs/dataset-notes.md).

## Verification

No training or application server has been run on the user's local computer.
Ten tests and the notebook's core workflow passed in a separate cloud sandbox
using synthetic fixtures only. Notebook JSON, source embedding, and packaging
are checked during authoring. The notebook repeats verification in Colab.
Actual dataset scores and Colab browser/runtime confirmation come from that run.
See [verification notes](docs/verification.md).

After editing source files, rebuild embedded notebooks with the PowerShell
file-authoring script. It writes notebooks without running the application.


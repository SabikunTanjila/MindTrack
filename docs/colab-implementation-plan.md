# MindTrack Colab implementation plan

**Goal:** Complete the existing AI lab workflow in a hosted Colab CPU runtime.
**Spec:** [colab-design.md](colab-design.md)
**Architecture:** Shared Python ML modules, a FastAPI API, an embedded web
dashboard, and notebooks with all required source embedded for direct upload.
**Tech stack:** Python, pandas, scikit-learn, Matplotlib, seaborn, FastAPI, Uvicorn,
HTML/CSS/JavaScript, pytest, and the Google Colab output API.

## Constraints

- Preserve Low, Medium, High, and Very High.
- Use the provided CSV; do not substitute the other files visible in the IDE.
- The delivered runtime is Colab. No project execution on the user's computer;
  isolated cloud verification uses synthetic fixtures only.
- Do not fit learned transformations on validation/test data.
- Do not use Mental_Health_Score as a predictor.
- Do not invent dataset provenance, performance, or clinical validity.

## Ordered tasks

- [x] Data contracts and tests: `ml/config.py`, `ml/preprocessing.py`,
  `tests/test_data.py`. Implement `prepare_data(frame) -> (frame, audit)` and
  `split_data(frame) -> dict` with disjoint stratified partitions.
- [x] Training: `ml/train.py`, `ml/clustering.py`, `ml/evaluate.py`,
  `ml/reporting.py`. Implement `train_classifiers(splits, quick=False) -> dict`,
  `train_clusters(X_train) -> dict`, `evaluate_classifiers(training, splits)`.
- [x] Inference: `backend/schemas.py`, `backend/predictor.py`,
  `backend/recommendations.py`, `backend/main.py`, `tests/test_integration.py`.
  Serialize via `export_artifacts(...)`; load through `Predictor(path)`;
  expose `create_app(model_dir)` with health, info, metrics, prediction, and UI.
- [x] Dashboard: `frontend/index.html`, `frontend/src/styles/main.css`,
  `frontend/src/services/app.js`. Build assessment/results/model-comparison views,
  loading/errors, input validation, probability bars, and cluster descriptions.
- [x] Notebook workflow: `scripts/build-notebooks.ps1` generates the main notebook
  and four topic notebooks from the same source. Every code cell has a numbered
  step and runs in Colab. Include tests, API smoke checks, iframe, and export.
- [x] Documentation/package: update README, dataset notes, requirements, Colab
  guide, and verification record. Inspect notebook JSON/embedded files and archive
  contents. Leave runtime evaluation explicitly pending until Colab executes it.

## Colab verification commands

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], check=True)
```

The integration test trains on a small fixture, reloads artifacts, and calls
the actual API via FastAPI TestClient. The full notebook then evaluates the
provided dataset and calls the running Colab API with a valid and invalid request.

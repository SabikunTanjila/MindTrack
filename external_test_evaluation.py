"""External evaluation for a pre-trained MindTrack model.

This script intentionally uses only the already trained model and its fitted
preprocessing objects. It does not modify the training methodology or tune on the
external test dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from ml.config import FEATURES, LABELS, TARGET


def load_and_validate_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"External dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(file_path)
    elif suffix in {'.xlsx', '.xls'}:
        frame = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported dataset format: {file_path.suffix}. Use .csv, .xlsx, or .xls.")

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame.empty:
        raise ValueError("External dataset is empty.")

    # Validate missing values before modeling.
    missing_values = frame.isna().sum()
    bad = missing_values[missing_values > 0]
    if not bad.empty:
        bad_cols = bad[bad > 0].to_dict()
        raise ValueError(f"Missing values detected in external dataset: {bad_cols}")

    # Validate target labels only as a safe guard; do not silently transform labels.
    actual_labels = sorted(frame[TARGET].dropna().astype(str).unique().tolist())
    unknown = [label for label in actual_labels if label not in LABELS]
    if unknown:
        raise ValueError(f"Unexpected Stress_Level values in external dataset: {unknown}. Expected: {LABELS}")

    # Validate numeric columns.
    for column in [c for c in FEATURES if c in frame.columns and c in ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']]:
        numeric = pd.to_numeric(frame[column], errors='coerce')
        if numeric.isna().any():
            bad_rows = int(numeric.isna().sum())
            raise ValueError(f"Invalid numeric values detected in column '{column}' ({bad_rows} rows)")

    # Category consistency check.
    for column in ['Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Stress_Level']:
        values = frame[column].astype(str).str.strip()
        if values.eq('').any():
            raise ValueError(f"Empty category values detected in column '{column}'")

    return frame


def load_project_artifact(model_path: str | Path):
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and 'models' in artifact and 'metadata' in artifact:
        return artifact
    raise TypeError(f"Unexpected artifact type in {model_path}. Expected a trained MindTrack bundle.")


def build_prediction_table(actual: pd.Series, predicted: pd.Series, probabilities: np.ndarray | None = None):
    result = pd.DataFrame({
        'Actual_Stress_Level': actual.astype(str).tolist(),
        'Predicted_Stress_Level': predicted.astype(str).tolist(),
        'Correct': (actual.astype(str) == predicted.astype(str)).tolist(),
    })
    if probabilities is not None:
        proba_cols = [f'Prob_{label}' for label in LABELS]
        for idx, label in enumerate(LABELS):
            result[f'Prob_{label}'] = probabilities[:, idx]
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained MindTrack model on an external Excel test dataset.')
    parser.add_argument('--data', required=True, help='Path to the external Excel or CSV dataset.')
    parser.add_argument('--artifact', default='models/mindtrack.joblib', help='Path to the saved trained artifact.')
    parser.add_argument('--output-dir', default='reports/external_test', help='Directory to store predictions and metrics.')
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_and_validate_dataset(args.data)
    bundle = load_project_artifact(artifact_path)

    model_bundle = bundle['models']
    primary_model_name = bundle['metadata'].get('primary_model', None)
    if primary_model_name is None:
        raise ValueError("Saved artifact does not contain the selected primary model information.")
    if primary_model_name not in model_bundle:
        raise ValueError(f"Primary model '{primary_model_name}' is missing from the saved artifact.")

    model = model_bundle[primary_model_name]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(frame[FEATURES])
    else:
        probabilities = None

    # The project expects preprocessing to be applied using the fitted pipeline that was
    # learned during training. This part reuses the learned preprocessing object from the
    # saved artifact, not a new fit on the external dataset.
    if 'preprocess' in model.named_steps:
        X = model.named_steps['preprocess'].transform(frame[FEATURES])
    else:
        X = frame[FEATURES]

    y_true = frame[TARGET].astype(str)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    metrics = {
        'label': 'External Test Dataset Evaluation',
        'num_samples': int(len(frame)),
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': LABELS,
    }

    result_table = build_prediction_table(y_true, pd.Series(y_pred, index=frame.index), probabilities)
    result_table.to_csv(output_dir / 'external_test_predictions.csv', index=False)
    with (output_dir / 'external_test_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, default=str)

    print('\nExternal Test Dataset Evaluation')
    print(f'Number of test samples: {len(frame)}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1-score (macro): {f1:.4f}')
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=LABELS, zero_division=0))
    print('\nConfusion matrix:')
    print(cm)
    print(f'\nPrediction results saved to: {output_dir / "external_test_predictions.csv"}')
    print(f'Metrics saved to: {output_dir / "external_test_metrics.json"}')


if __name__ == '__main__':
    main()

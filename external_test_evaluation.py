"""Evaluate a fitted MindTrack pipeline on a held-out CSV or Excel dataset.

The external data is never used to refit preprocessing or select a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from ml.config import BOUNDS, CATEGORICAL, FEATURES, LABELS, NUMERIC, TARGET

AGE_BINS = [12, 17, 24, 34, 44, 59, 120]
AGE_GROUPS = ['13-17', '18-24', '25-34', '35-44', '45-59', '60-120']


def load_and_validate_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f'External dataset not found: {file_path}')
    if file_path.suffix.lower() == '.csv':
        frame = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {'.xlsx', '.xls'}:
        frame = pd.read_excel(file_path)
    else:
        raise ValueError(f'Unsupported dataset format: {file_path.suffix}. Use .csv, .xlsx, or .xls.')
    return validate_dataframe(frame)


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame.empty:
        raise ValueError('External dataset is empty.')
    frame = frame.copy()
    missing_values = frame[FEATURES + [TARGET]].isna().sum()
    if missing_values.any():
        raise ValueError(f'Missing values detected in external dataset: {missing_values[missing_values > 0].to_dict()}')

    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
        if frame[column].isna().any():
            raise ValueError(f"Invalid numeric values detected in column '{column}' ({int(frame[column].isna().sum())} rows)")
        low, high = BOUNDS[column]
        outside = ~frame[column].between(low, high)
        if outside.any():
            raise ValueError(f"Values outside the supported range {low}-{high} in column '{column}' ({int(outside.sum())} rows)")

    for column in CATEGORICAL + [TARGET]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq('').any():
            raise ValueError(f"Empty category values detected in column '{column}'")
    unknown = sorted(set(frame[TARGET]) - set(LABELS))
    if unknown:
        raise ValueError(f'Unexpected {TARGET} values: {unknown}. Expected: {LABELS}')
    return frame


def load_project_artifact(model_path: str | Path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Model artifact not found: {path}. Run the training/export cells before external evaluation.')
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {'models', 'metadata'} <= artifact.keys():
        raise TypeError(f'Unexpected artifact type in {path}. Expected a trained MindTrack bundle.')
    if artifact['metadata'].get('artifact_version') != 2:
        raise ValueError('This artifact predates broad-age support. Rerun training and export it before evaluation.')
    return artifact


def selected_model(bundle):
    name = bundle['metadata'].get('primary_model')
    if not name:
        raise ValueError('Saved artifact does not identify the selected primary model.')
    if name not in bundle['models']:
        raise ValueError(f"Primary model '{name}' is missing from the saved artifact.")
    return name, bundle['models'][name]


def build_prediction_table(actual, predicted, probabilities=None, probability_classes=None):
    actual_values = pd.Series(actual).astype(str).to_numpy()
    predicted_values = np.asarray(predicted).astype(str)
    result = pd.DataFrame({
        'Actual_Stress_Level': actual_values,
        'Predicted_Stress_Level': predicted_values,
        'Correct': actual_values == predicted_values,
    })
    if probabilities is not None:
        probability_classes = [str(value) for value in probability_classes]
        class_index = {label: index for index, label in enumerate(probability_classes)}
        for label in LABELS:
            result[f'Prob_{label}'] = probabilities[:, class_index[label]]
    return result


def metric_summary(actual, predicted):
    return {
        'num_samples': int(len(actual)),
        'accuracy': float(accuracy_score(actual, predicted)),
        'precision_macro': float(precision_score(actual, predicted, labels=LABELS, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(actual, predicted, labels=LABELS, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(actual, predicted, labels=LABELS, average='macro', zero_division=0)),
    }


def evaluate_bundle(bundle, frame):
    frame = validate_dataframe(frame)
    model_name, model = selected_model(bundle)
    raw_features = frame[FEATURES]
    # A fitted sklearn Pipeline owns its preprocessing. Passing transformed data
    # here would apply FeatureCleaner/ColumnTransformer twice and break inference.
    predicted = model.predict(raw_features)
    probabilities = model.predict_proba(raw_features) if hasattr(model, 'predict_proba') else None
    actual = frame[TARGET].astype(str).to_numpy()
    metrics = metric_summary(actual, predicted)
    metrics.update({
        'label': 'External Test Dataset Evaluation',
        'primary_model': model_name,
        'classification_report': classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        'labels': LABELS,
        'age_policy': bundle['metadata'].get('limitations', []),
    })
    age_groups = pd.cut(frame['Age'], bins=AGE_BINS, labels=AGE_GROUPS, include_lowest=True)
    metrics['metrics_by_age_group'] = {}
    for group in AGE_GROUPS:
        mask = age_groups.astype('string').eq(group).fillna(False).to_numpy()
        if mask.any():
            metrics['metrics_by_age_group'][group] = metric_summary(actual[mask], np.asarray(predicted)[mask])
    predictions = build_prediction_table(actual, predicted, probabilities,
                                         getattr(model, 'classes_', None))
    predictions.insert(0, 'Age', frame['Age'].to_numpy())
    predictions.insert(1, 'Age_Group', age_groups.astype('string').to_numpy())
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained MindTrack model on external data.')
    parser.add_argument('--data', required=True, help='Path to an external CSV, XLSX, or XLS dataset.')
    parser.add_argument('--artifact', default='models/mindtrack.joblib', help='Path to the trained artifact.')
    parser.add_argument('--output-dir', default='reports/external_test', help='Directory for predictions and metrics.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_and_validate_dataset(args.data)
    bundle = load_project_artifact(args.artifact)
    metrics, predictions = evaluate_bundle(bundle, frame)
    predictions.to_csv(output_dir / 'external_test_predictions.csv', index=False)
    (output_dir / 'external_test_metrics.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')

    print('\nExternal Test Dataset Evaluation')
    print(f"Primary model: {metrics['primary_model']}")
    print(f"Number of test samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    if metrics['metrics_by_age_group']:
        print('\nMetrics by age group:')
        for group, values in metrics['metrics_by_age_group'].items():
            print(f"  {group}: n={values['num_samples']}, F1={values['f1_macro']:.4f}")
    print(f"\nPrediction results saved to: {output_dir / 'external_test_predictions.csv'}")
    print(f"Metrics saved to: {output_dir / 'external_test_metrics.json'}")


if __name__ == '__main__':
    main()

"""Reproducible charts, evaluation tables, and self-contained model artifacts."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from ml.config import (BOUNDS, CATEGORICAL, DISPLAY_NAMES, EXCLUDED, FEATURES,
                       LABELS, NUMERIC, SEED, TARGET)


def save_json(path, value):
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [convert(v) for v in obj]
        if isinstance(obj, np.generic):
            return convert(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(convert(value), indent=2, allow_nan=False), encoding='utf-8')


def plot_eda(frame, directory):
    """Call on the training partition; full-file inspection is limited to quality audit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', palette='deep')
    figures = []
    fig, ax = plt.subplots(figsize=(7, 4))
    frame[TARGET].value_counts().reindex(LABELS, fill_value=0).plot.bar(ax=ax, color='#227d75')
    ax.set(title='Training split: stress labels', ylabel='Records', xlabel='Dataset label')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(directory / 'class_distribution.png', dpi=150)
    figures.append(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, ax in zip(NUMERIC, axes.flat):
        sns.histplot(frame[column], bins=20, ax=ax, color='#227d75')
        ax.set_title(DISPLAY_NAMES[column])
    fig.tight_layout()
    fig.savefig(directory / 'feature_distributions.png', dpi=150)
    figures.append(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(frame[NUMERIC].corr(), annot=True, fmt='.2f', vmin=-1, vmax=1,
                cmap='vlag', ax=ax)
    ax.set_title('Training feature correlations (association, not causation)')
    fig.tight_layout()
    fig.savefig(directory / 'correlations.png', dpi=150)
    figures.append(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, ax in zip(NUMERIC, axes.flat):
        sns.boxplot(data=frame, x=TARGET, y=column, order=LABELS, ax=ax, color='#91c6b6')
        ax.set_title(DISPLAY_NAMES[column])
    fig.tight_layout()
    fig.savefig(directory / 'features_by_stress.png', dpi=150)
    figures.append(fig)
    return figures


def plot_clusters(clustering, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores = clustering['scores']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(scores.k, scores.inertia, 'o-', color='#227d75')
    axes[0].set(title='Elbow plot', xlabel='Cluster count k', ylabel='Inertia')
    axes[1].plot(scores.k, scores.silhouette, 'o-', color='#227d75')
    axes[1].set(title='Silhouette comparison', xlabel='Cluster count k', ylabel='Silhouette')
    points = clustering['coordinates']
    sns.scatterplot(data=points, x='PC1', y='PC2', hue=points['cluster'].astype(str),
                    alpha=.5, s=15, ax=axes[2])
    variance = clustering['pca'].explained_variance_ratio_.sum()
    axes[2].set_title(f'Training PCA: {variance:.1%} variance shown')
    fig.tight_layout()
    fig.savefig(directory / 'clustering.png', dpi=150)
    return fig


def plot_evaluation(evaluation, importance, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (name, result), ax in zip(evaluation.items(), axes.flat):
        ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=LABELS).plot(
            ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(directory / 'confusion_matrices.png', dpi=150)
    fig2, ax = plt.subplots(figsize=(9, 5))
    values = importance.sort_values('importance')
    ax.barh(values.feature.map(DISPLAY_NAMES), values.importance, xerr=values['std'], color='#227d75')
    ax.set(title='Global permutation importance on validation data',
           xlabel='Drop in macro F1 after shuffling a feature')
    fig2.tight_layout()
    fig2.savefig(directory / 'global_importance.png', dpi=150)
    return fig, fig2


def metric_table(evaluation):
    return pd.DataFrame([{key: value for key, value in result.items()
                          if key not in ('confusion_matrix', 'classification_report')} | {'model': name}
                         for name, result in evaluation.items()]).set_index('model')


def export_artifacts(training, clustering, evaluation, importance, splits, audit,
                     output_dir, dataset_path=None):
    output_dir = Path(output_dir)
    model_dir = output_dir / 'models'
    metrics_dir = output_dir / 'reports' / 'metrics'
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train = splits['X_train']
    metadata = {
        'artifact_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'labels': LABELS, 'features': FEATURES,
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'display_names': DISPLAY_NAMES, 'bounds': BOUNDS, 'excluded_columns': EXCLUDED + [TARGET],
        'primary_model': training['primary_model'], 'selection_rule': 'Highest validation macro F1',
        'selection': training['selection'].to_dict(orient='records'),
        'categories': {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL},
        'medians': train[NUMERIC].median().to_dict(),
        'training_ranges': {c: [float(train[c].min()), float(train[c].max())] for c in NUMERIC},
        'split_sizes': {s: len(splits[f'X_{s}']) for s in ('train', 'val', 'test')},
        'cluster_count': clustering['best_k'], 'cluster_descriptions': clustering['descriptions'],
        'cluster_profiles': clustering['profiles'].reset_index().to_dict(orient='records'),
        'cluster_scores': clustering['scores'].to_dict(orient='records'),
        'global_importance': importance.to_dict(orient='records'),
        'dataset_audit': audit, 'python_version': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'joblib', 'fastapi', 'pydantic']},
        'limitations': [
            'Educational estimates of dataset labels, not a clinical assessment.',
            'Dataset source, license, collection method, and synthetic status are unverified.',
            'The dataset covers ages 18–24; performance outside this cohort is untested.',
            'Probabilities are uncalibrated model outputs.',
            'Study and physical-activity time windows need source confirmation.',
            'Global feature importance and input comparisons do not establish individual causes.',
        ],
    }
    if dataset_path:
        metadata['dataset_sha256'] = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    bundle = {'models': training['models'], 'cluster_pipeline': clustering['pipeline'],
              'pca': clustering['pca'], 'metadata': metadata, 'evaluation': evaluation}
    joblib.dump(bundle, model_dir / 'mindtrack.joblib', compress=3)
    save_json(model_dir / 'metadata.json', metadata)
    save_json(metrics_dir / 'classification.json', evaluation)
    save_json(metrics_dir / 'data_audit.json', audit)
    metric_table(evaluation).to_csv(metrics_dir / 'classification.csv')
    training['selection'].to_csv(metrics_dir / 'model_selection.csv', index=False)
    clustering['scores'].to_csv(metrics_dir / 'clustering.csv', index=False)
    clustering['profiles'].to_csv(metrics_dir / 'cluster_profiles.csv')
    importance.to_csv(metrics_dir / 'global_importance.csv', index=False)
    for name, search in training['searches'].items():
        search.to_csv(metrics_dir / f'cv_{name.lower().replace(" ", "_")}.csv', index=False)
    from ml.writeup import generate_writeup
    generate_writeup(metadata, evaluation, output_dir)
    return model_dir / 'mindtrack.joblib'

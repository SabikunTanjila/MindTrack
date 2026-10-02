"""Train and export the MindTrack model bundle for local use."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_dataset() -> Path:
    """Return the first supported local dataset location."""
    candidates = (
        ROOT / 'data' / 'raw' / 'mental_health_dataset.csv',
        ROOT / 'mental_health_dataset.csv',
    )
    for path in candidates:
        if path.is_file():
            return path
    locations = '\n'.join(f'  - {path}' for path in candidates)
    raise FileNotFoundError(
        'The training dataset was not found. Put mental_health_dataset.csv in one of:\n'
        f'{locations}'
    )


def train_local(quick: bool = False) -> Path:
    """Train, evaluate, and write ``models/mindtrack.joblib``."""
    try:
        import pandas as pd

        from ml.clustering import train_clusters
        from ml.evaluate import evaluate_classifiers, global_importance
        from ml.preprocessing import prepare_data, split_data
        from ml.reporting import export_artifacts
        from ml.train import train_classifiers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f'Missing Python dependency: {exc.name}. Install dependencies with '
            f'"{sys.executable} -m pip install -r requirements.txt".'
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            'A scientific Python package could not be loaded. Recreate the virtual '
            'environment and reinstall requirements; copied or stale virtual '
            f'environments are not portable. Original error: {exc}'
        ) from exc

    dataset_path = find_dataset()
    print(f'[1/6] Loading dataset: {dataset_path}')
    frame, audit = prepare_data(pd.read_csv(dataset_path))

    print('[2/6] Validating, cleaning, and splitting the data...')
    splits = split_data(frame)

    mode = 'quick' if quick else 'full'
    print(f'[3/6] Training classifiers ({mode} search)...')
    training = train_classifiers(splits, quick=quick)
    print(f"      Primary model: {training['primary_model']}")

    print('[4/6] Training lifestyle clusters...')
    clustering = train_clusters(splits['X_train'])

    print('[5/6] Evaluating the fitted pipelines...')
    evaluation = evaluate_classifiers(training, splits)
    importance = global_importance(training, splits, repeats=2 if quick else 5)

    print('[6/6] Exporting the model bundle...')
    model_path = export_artifacts(
        training=training,
        clustering=clustering,
        evaluation=evaluation,
        importance=importance,
        splits=splits,
        audit=audit,
        output_dir=ROOT,
        dataset_path=dataset_path,
    )
    print(f'Model ready: {model_path}')
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the MindTrack model locally.')
    parser.add_argument(
        '--quick', action='store_true',
        help='Use one parameter combination per model for a faster development build.',
    )
    args = parser.parse_args()
    train_local(quick=args.quick)


if __name__ == '__main__':
    main()

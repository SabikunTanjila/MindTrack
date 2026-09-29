"""Train and export the MindTrack model bundle locally.

Uses mental_health_dataset.csv to train classifiers, fit K-Means clustering,
evaluate models, and export models/mindtrack.joblib.

Run with:
    python train_local.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.preprocessing import prepare_data, split_data
from ml.train import train_classifiers
from ml.clustering import train_clusters
from ml.evaluate import evaluate_classifiers, global_importance
from ml.reporting import export_artifacts


def train_local(quick: bool = False):
    dataset_path = ROOT / 'data' / 'raw' / 'mental_health_dataset.csv'
    if not dataset_path.exists():
        dataset_path = ROOT / 'mental_health_dataset.csv'

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"[1/6] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("[2/6] Cleaning data and validating schema...")
    frame, audit = prepare_data(df)

    print("[3/6] Creating stratified 60/20/20 train/validation/test splits...")
    splits = split_data(frame)

    print("[4/6] Training & tuning classifiers (KNN, Logistic Regression, Random Forest)...")
    training = train_classifiers(splits, quick=quick)
    print(f"      Selected primary model: {training['primary_model']}")

    print("[5/6] Fitting K-Means clustering & PCA...")
    clustering = train_clusters(splits['X_train'])

    print("[6/6] Evaluating on holdout test set & exporting artifacts...")
    evaluation = evaluate_classifiers(training, splits)
    importance = global_importance(training, splits, repeats=5 if not quick else 2)

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
    print("=" * 60)
    print(f"SUCCESS: Model bundle generated at:\n  {model_path}")
    print("You can now start the server with: python run.py")
    print("=" * 60)
    return model_path


if __name__ == '__main__':
    train_local()

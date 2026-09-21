"""Holdout evaluation and aggregate explanations, with explicit label ordering."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from ml.config import FEATURES, LABELS, SEED


def classification_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = model.classes_
    auc = None
    if set(y) == set(classes):
        auc = float(roc_auc_score(y, probabilities, labels=classes, multi_class='ovr', average='macro'))
    return {'accuracy': float(accuracy_score(y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
            'precision_macro': float(precision_score(y, predictions, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y, predictions, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y, predictions, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y, predictions, average='weighted', zero_division=0)),
            'roc_auc_ovr_macro': auc,
            'confusion_matrix': confusion_matrix(y, predictions, labels=LABELS).tolist(),
            'classification_report': classification_report(y, predictions, labels=LABELS, output_dict=True, zero_division=0)}


def evaluate_classifiers(training, splits):
    models = dict(training['models']) | {'Majority baseline': training['baseline']}
    return {name: classification_metrics(model, splits['X_test'], splits['y_test'])
            for name, model in models.items()}


def global_importance(training, splits, repeats=5):
    model = training['models'][training['primary_model']]
    result = permutation_importance(model, splits['X_val'], splits['y_val'],
                                    scoring='f1_macro', n_repeats=repeats, random_state=SEED, n_jobs=2)
    return pd.DataFrame({'feature': FEATURES, 'importance': result.importances_mean,
                         'std': result.importances_std}).sort_values('importance', ascending=False)

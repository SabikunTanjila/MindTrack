"""Deterministic cleaning followed by transformations learned within each split."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml.config import BOUNDS, CATEGORICAL, FEATURES, LABELS, NUMERIC, SEED, TARGET


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Stateless rules are identical during training and inference."""
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(self.features or FEATURES, dtype=object)
        return self

    def transform(self, X):
        columns = self.features or FEATURES
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f'Missing feature columns: {missing}')
        result = X.loc[:, columns].copy()
        for column in columns:
            if column in NUMERIC:
                values = pd.to_numeric(result[column], errors='coerce')
                lo, hi = BOUNDS[column]
                result[column] = values.where(np.isfinite(values) & values.between(lo, hi))
            else:
                values = result[column].astype('string').str.strip()
                result[column] = values.replace('', pd.NA).astype(object)
                result[column] = result[column].where(pd.notna(result[column]), np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features or FEATURES, dtype=object)


def make_preprocessor(features=None):
    columns = features or FEATURES
    numeric = [c for c in columns if c in NUMERIC]
    categorical = [c for c in columns if c in CATEGORICAL]
    transformers = [('numeric', Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ]), numeric)]
    if categorical:
        transformers.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ('clean', FeatureCleaner(features=columns)),
        ('columns', ColumnTransformer(transformers, remainder='drop')),
    ])


def prepare_data(frame):
    """Audit and deduplicate without learning statistics from the complete dataset."""
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    audit = {'raw_rows': len(frame), 'raw_columns': list(frame.columns),
             'raw_missing': frame.isna().sum().astype(int).to_dict(),
             'exact_duplicates_removed': int(frame.duplicated().sum())}
    frame = frame.drop_duplicates().copy()
    target = frame[TARGET].astype('string').str.strip().replace('', pd.NA)
    unknown = sorted(set(target.dropna()) - set(LABELS))
    if unknown:
        raise ValueError(f'Unknown stress labels: {unknown}. Expected {LABELS}.')
    audit['missing_target_rows_removed'] = int(target.isna().sum())
    keep = target.notna()
    frame, target = frame.loc[keep], target.loc[keep]
    clean = FeatureCleaner().transform(frame)
    audit['invalid_numeric_values'] = {}
    for column in NUMERIC:
        audit['invalid_numeric_values'][column] = int(
            (frame[column].notna() & clean[column].isna()).sum())
    clean[TARGET] = target.astype(str)
    # Predictor duplicates must not leak across partitions, including conflicts.
    groups = clean.groupby(FEATURES, dropna=False, sort=False)[TARGET].transform('nunique')
    conflicts = groups > 1
    audit['conflicting_profile_rows_removed'] = int(conflicts.sum())
    clean = clean.loc[~conflicts].copy()
    repeated = clean.duplicated(subset=FEATURES)
    audit['repeated_profile_rows_removed'] = int(repeated.sum())
    clean = clean.loc[~repeated].reset_index(drop=True)
    audit['prepared_rows'] = len(clean)
    audit['class_counts'] = clean[TARGET].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    audit['remaining_missing'] = clean.isna().sum().astype(int).to_dict()
    return clean, audit


def split_data(frame):
    counts = frame[TARGET].value_counts().reindex(LABELS, fill_value=0)
    if counts.min() < 15:
        raise ValueError('Need at least 15 unique profiles per class for stratified splitting and CV.')
    X, y = frame[FEATURES], frame[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)
    empty = X_train.columns[X_train.isna().all()].tolist()
    if empty:
        raise ValueError(f'No usable training values for columns: {empty}')
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test)

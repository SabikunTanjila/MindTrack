"""Classifier selection uses CV and validation; the test split stays untouched."""
from time import perf_counter
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from ml.config import SEED
from ml.preprocessing import make_preprocessor


def train_classifiers(splits, quick=False):
    configurations = {
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7, 9],
                                         'model__weights': ['uniform', 'distance']}),
        'Logistic Regression': (LogisticRegression(max_iter=3000, random_state=SEED),
                                {'model__C': [0.1, 1.0, 10.0]}),
        'Random Forest': (RandomForestClassifier(random_state=SEED, n_jobs=1),
                          {'model__n_estimators': [100, 200], 'model__max_depth': [None, 12],
                           'model__min_samples_leaf': [1, 3]}),
    }
    fitted, rows, searches = {}, [], {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for name, (estimator, parameters) in configurations.items():
        if quick:
            parameters = {key: [values[0]] for key, values in parameters.items()}
        pipeline = Pipeline([('preprocess', make_preprocessor()), ('model', estimator)])
        search = GridSearchCV(pipeline, parameters, scoring='f1_macro', cv=cv,
                              n_jobs=2, refit=True, error_score='raise')
        start = perf_counter()
        search.fit(splits['X_train'], splits['y_train'])
        seconds = perf_counter() - start
        predicted = search.predict(splits['X_val'])
        fitted[name] = search.best_estimator_
        searches[name] = pd.DataFrame(search.cv_results_)
        rows.append({'model': name, 'cv_macro_f1': float(search.best_score_),
                     'validation_macro_f1': float(f1_score(splits['y_val'], predicted, average='macro')),
                     'fit_seconds': seconds, 'best_parameters': search.best_params_})
        print(f'{name}: validation macro F1 = {rows[-1]["validation_macro_f1"]:.4f}', flush=True)
    selection = pd.DataFrame(rows).sort_values('validation_macro_f1', ascending=False, kind='stable')
    primary = str(selection.iloc[0]['model'])
    baseline = Pipeline([('preprocess', make_preprocessor()), ('model', DummyClassifier(strategy='most_frequent'))])
    baseline.fit(splits['X_train'], splits['y_train'])
    return {'models': fitted, 'primary_model': primary, 'selection': selection,
            'searches': searches, 'baseline': baseline}

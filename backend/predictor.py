"""Reload complete fitted pipelines; never retrain or refit on assessment input."""
from pathlib import Path

import joblib
import pandas as pd

from backend.recommendations import suggestions
from backend.schemas import Assessment
from ml.config import CATEGORY_ALIASES, DISPLAY_NAMES, LABELS, NUMERIC
from ml.recommendation import DEFAULT_THRESHOLDS


class Predictor:
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                'Model bundle not found. Run "python train_local.py" locally or export it from Colab.'
            )
        bundle = joblib.load(path)
        if bundle['metadata'].get('artifact_version') != 2:
            raise ValueError('Unsupported artifact version. Rerun training and export for broad-age support.')
        self.models = bundle['models']
        self.cluster_pipeline = bundle['cluster_pipeline']
        self.metadata = bundle['metadata']
        self.evaluation = bundle['evaluation']
        self.labels = [str(label) for label in self.metadata.get('labels', LABELS)]
        self.recommendation_thresholds = self.metadata.get(
            'recommendation_thresholds', DEFAULT_THRESHOLDS,
        )

    def predict(self, assessment):
        if not isinstance(assessment, Assessment):
            assessment = Assessment.model_validate(assessment)
        values = assessment.model_dump()
        frame = pd.DataFrame([values])
        results = []
        for name, pipeline in self.models.items():
            scores = dict(zip(pipeline.classes_, pipeline.predict_proba(frame)[0]))
            label = str(pipeline.predict(frame)[0])
            results.append({'model': name, 'label': label,
                            'probabilities': {key: float(scores[key]) for key in self.labels}})
        primary = next(r for r in results if r['model'] == self.metadata['primary_model'])
        cluster_id = int(self.cluster_pipeline.predict(frame)[0])
        comparisons = [{'feature': column, 'name': DISPLAY_NAMES[column], 'value': values[column],
                        'training_median': float(self.metadata['medians'][column])}
                       for column in NUMERIC]
        notes = []
        age_low, age_high = self.metadata['training_ranges']['Age']
        if not age_low <= values['Age'] <= age_high:
            notes.append(
                f'Age is outside the observed training range ({age_low:g}-{age_high:g}). '
                'Age is excluded from prediction, but performance for this age group is unverified.')
        for column in NUMERIC[1:]:
            low, high = self.metadata['training_ranges'][column]
            if not low <= values[column] <= high:
                notes.append(f'{DISPLAY_NAMES[column]} is outside the observed training range ({low:g}-{high:g}); this prediction may be less reliable.')
        for column, choices in self.metadata['categories'].items():
            canonical = CATEGORY_ALIASES.get(column, {}).get(values[column], values[column])
            if canonical not in choices:
                notes.append(f'{DISPLAY_NAMES[column]} was unseen in training and is encoded as an unknown category.')
        return {'label': primary['label'], 'primary_model': primary['model'],
                'probabilities': primary['probabilities'], 'models': results,
                'agreement': sum(r['label'] == primary['label'] for r in results),
                'cluster': {'id': cluster_id, 'description': self.metadata['cluster_descriptions'][str(cluster_id)]},
                'comparisons': comparisons, 'suggestions': suggestions(values, self.metadata['medians']),
                'notes': notes, 'limitations': self.metadata['limitations']}

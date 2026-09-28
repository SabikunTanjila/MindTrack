"""Small synthetic contract fixture; not evidence of real-dataset accuracy."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.predictor import Predictor
from ml.clustering import train_clusters
from ml.evaluate import evaluate_classifiers, global_importance
from ml.preprocessing import prepare_data, split_data
from ml.reporting import export_artifacts
from ml.train import train_classifiers
from external_test_evaluation import evaluate_bundle


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    rng = np.random.default_rng(123)
    records=[]
    for i in range(160):
        band = i % 4
        records.append(dict(Age=18+i%7, Avg_Daily_Usage_Hours=1+band*1.5+rng.uniform(0,.4),
            Daily_Unlocks=70+i, Study_Hours=7-band+rng.uniform(0,.4),
            Physical_Activity_Hours=3-band*.4, Sleep_Hours_Per_Night=8-band*.8,
            Academic_Level='Undergraduate', Most_Used_Platform='Facebook', Purpose_Of_Use='Networking',
            Stress_Level=['Low','Medium','High','Very High'][band]))
    frame, audit = prepare_data(pd.DataFrame(records))
    splits = split_data(frame)
    training = train_classifiers(splits, quick=True)
    clustering = train_clusters(splits['X_train'], k_values=range(2,4))
    evaluation = evaluate_classifiers(training, splits)
    importance = global_importance(training, splits, repeats=2)
    root = tmp_path_factory.mktemp('artifacts')
    path = export_artifacts(training, clustering, evaluation, importance, splits, audit, root)
    payload = records[7].copy(); payload.pop('Stress_Level')
    return root, path, payload, training, evaluation


def test_serialization_retains_predictions_and_probability_labels(trained):
    root, path, payload, training, evaluation = trained
    predictor = Predictor(path)
    result = predictor.predict(payload)
    expected = training['models'][training['primary_model']].predict(pd.DataFrame([payload]))[0]
    assert result['label'] == expected
    assert set(result['probabilities']) == {'Low', 'Medium', 'High', 'Very High'}
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    assert result['probabilities'][result['label']] == max(result['probabilities'].values())
    assert len(result['models']) == 3
    assert len(evaluation) == 4
    assert np.array(evaluation['KNN']['confusion_matrix']).sum() == 32


def test_api_validates_inputs_and_serves_dashboard(trained):
    root, path, payload, *_ = trained
    with TestClient(create_app(root/'models')) as client:
        assert client.get('/health').json()['models_loaded'] is True
        assert client.get('/').status_code == 200
        assert client.get('/static/services/app.js').status_code == 200
        assert client.get('/metrics').status_code == 200
        assert client.post('/predict', json=payload).status_code == 200
        for age in (15, 32, 72):
            response = client.post('/predict', json=payload | {'Age': age})
            assert response.status_code == 200
            assert response.json()['label'] == client.post('/predict', json=payload).json()['label']
        for changes in [{'Age': 12}, {'Sleep_Hours_Per_Night': -1}, {'Daily_Unlocks': 2.5},
                        {'Mental_Health_Score': 8}, {'Most_Used_Platform': '   '}]:
            assert client.post('/predict', json=payload | changes).status_code == 422
        missing = dict(payload); missing.pop('Study_Hours')
        assert client.post('/predict', json=missing).status_code == 422


def test_unseen_platform_warns_without_breaking_inference(trained):
    _, path, payload, *_ = trained
    result = Predictor(path).predict(payload | {'Most_Used_Platform': 'New platform'})
    assert any('unseen' in note for note in result['notes'])


def test_external_evaluation_uses_raw_pipeline_and_correct_probability_labels(trained):
    _, _, payload, training, _ = trained
    primary = training['models'][training['primary_model']]
    rows = []
    for age, label in [(15, 'Low'), (32, 'Medium'), (72, 'High')]:
        rows.append(payload | {'Age': age, 'Stress_Level': label})
    frame = pd.DataFrame(rows)
    bundle = {
        'models': training['models'],
        'metadata': {
            'primary_model': training['primary_model'],
            'artifact_version': 2,
            'limitations': [],
        },
    }
    metrics, predictions = evaluate_bundle(bundle, frame)
    direct = primary.predict(frame.drop(columns='Stress_Level'))
    assert predictions['Predicted_Stress_Level'].tolist() == direct.tolist()
    expected = dict(zip(primary.classes_, primary.predict_proba(frame.drop(columns='Stress_Level'))[0]))
    for label, probability in expected.items():
        assert predictions.loc[0, f'Prob_{label}'] == pytest.approx(probability)
    assert set(metrics['metrics_by_age_group']) == {'13-17', '25-34', '60-120'}


def test_missing_models_return_service_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/health').json()['models_loaded'] is False
        assert client.get('/model-info').status_code == 503

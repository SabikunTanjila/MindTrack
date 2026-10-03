"""Input-rule tests; recommendation thresholds are not clinical boundaries."""
import pandas as pd
import pytest

from ml.config import LABELS
from ml.recommendation import (DEFAULT_THRESHOLDS,
                               build_recommendation_thresholds,
                               generate_recommendations)


def assessment(**changes):
    values = {
        'Age': 30,
        'Academic_Level': 'Graduate',
        'Avg_Daily_Usage_Hours': 5.0,
        'Most_Used_Platform': 'YouTube',
        'Purpose_Of_Use': 'Education',
        'Daily_Unlocks': 171,
        'Study_Hours': 2.8,
        'Physical_Activity_Hours': 1.7,
        'Sleep_Hours_Per_Night': 6.6,
    }
    return values | changes


def category(result, name):
    return next(item for item in result['items'] if item['category'] == name)


def test_thresholds_are_calculated_from_training_frame_quantiles():
    frame = pd.DataFrame({
        'Sleep_Hours_Per_Night': [4, 6, 8, 10],
        'Avg_Daily_Usage_Hours': [2, 4, 6, 8],
        'Daily_Unlocks': [50, 100, 150, 200],
        'Physical_Activity_Hours': [0, 1, 2, 3],
        'Study_Hours': [1, 3, 5, 7],
    })
    rules = build_recommendation_thresholds(frame)
    assert rules['source'] == 'training_split_quantiles'
    assert rules['Sleep_Hours_Per_Night']['low'] == pytest.approx(5.5)
    assert rules['Avg_Daily_Usage_Hours']['high'] == pytest.approx(6.5)
    assert rules['Daily_Unlocks']['reference'] == pytest.approx(125)


@pytest.mark.parametrize('changes, expected', [
    ({'Sleep_Hours_Per_Night': 4.5}, 'Sleep'),
    ({'Avg_Daily_Usage_Hours': 8.0}, 'Screen time'),
    ({'Daily_Unlocks': 250}, 'Digital interruptions'),
    ({'Physical_Activity_Hours': 0.5}, 'Physical activity'),
    ({'Study_Hours': 7.0}, 'Study balance'),
])
def test_each_input_threshold_generates_its_rule(changes, expected):
    result = generate_recommendations(assessment(**changes), 'Low', LABELS)
    item = category(result, expected)
    assert item['reason']
    assert item['action']
    assert item['observed_value'] is not None
    assert item['threshold_value'] is not None


def test_screen_recommendation_is_personalized_with_existing_category_inputs():
    result = generate_recommendations(
        assessment(Avg_Daily_Usage_Hours=8, Most_Used_Platform='Instagram',
                   Purpose_Of_Use='Networking'),
        'Medium', LABELS,
    )
    item = category(result, 'Screen time')
    assert 'Instagram' in item['reason']
    assert 'Networking' in item['reason']


def test_recommendation_priority_matches_model_risk_level():
    low = generate_recommendations(assessment(Sleep_Hours_Per_Night=5.2), 'Low', LABELS)
    medium = generate_recommendations(assessment(Sleep_Hours_Per_Night=5.2), 'Medium', LABELS)
    high = generate_recommendations(assessment(Sleep_Hours_Per_Night=5.2), 'High', LABELS)
    very_high = generate_recommendations(assessment(Sleep_Hours_Per_Night=5.2), 'Very High', LABELS)
    assert category(low, 'Sleep')['priority'] == 'Low'
    assert category(medium, 'Sleep')['priority'] == 'Medium'
    assert category(high, 'Sleep')['priority'] == 'High'
    assert category(very_high, 'Sleep')['priority'] == 'High'
    assert 'General wellbeing' not in [item['category'] for item in low['items']]
    assert category(high, 'General wellbeing')['priority'] == 'High'
    assert 'not a diagnosis' in category(high, 'General wellbeing')['reason']


@pytest.mark.parametrize('risk, expected', [
    ('Low', 'Low'),
    ('Medium', 'Medium'),
    ('High', 'High'),
    ('Very High', 'High'),
])
def test_every_recommendation_uses_priority_for_predicted_risk(risk, expected):
    result = generate_recommendations(
        assessment(Sleep_Hours_Per_Night=4, Avg_Daily_Usage_Hours=9,
                   Daily_Unlocks=260, Study_Hours=8,
                   Physical_Activity_Hours=0),
        risk, LABELS,
    )
    assert {item['priority'] for item in result['items']} == {expected}


def test_visible_reasons_use_plain_language_not_statistical_terms():
    result = generate_recommendations(
        assessment(Sleep_Hours_Per_Night=4, Avg_Daily_Usage_Hours=9,
                   Daily_Unlocks=260, Study_Hours=8,
                   Physical_Activity_Hours=0),
        'High', LABELS,
    )
    visible_text = ' '.join(item['reason'] for item in result['items']).lower()
    for term in ('quartile', 'percentile', 'training-data', 'recording period'):
        assert term not in visible_text


def test_age_only_changes_support_wording_not_priority():
    younger = generate_recommendations(assessment(Age=15), 'Very High', LABELS)
    older = generate_recommendations(assessment(Age=72), 'Very High', LABELS)
    young_support = category(younger, 'General wellbeing')
    old_support = category(older, 'General wellbeing')
    assert young_support['priority'] == old_support['priority'] == 'High'
    assert 'trusted adult' in young_support['action']
    assert 'trusted adult' not in old_support['action']


def test_invalid_optional_values_do_not_crash_rule_engine():
    result = generate_recommendations(
        {'Sleep_Hours_Per_Night': None, 'Avg_Daily_Usage_Hours': 'unknown',
         'Daily_Unlocks': float('nan')},
        'Low', LABELS,
    )
    assert result['items'][0]['category'] == 'Routine balance'


def test_recommendations_are_unique_and_capped():
    result = generate_recommendations(
        assessment(Sleep_Hours_Per_Night=4, Avg_Daily_Usage_Hours=9,
                   Daily_Unlocks=260, Study_Hours=8,
                   Physical_Activity_Hours=0),
        'Very High', LABELS,
    )
    categories = [item['category'] for item in result['items']]
    assert len(result['items']) == 6
    assert len(categories) == len(set(categories))
    assert 'Sleep' in categories


def test_defaults_document_their_training_data_origin():
    assert DEFAULT_THRESHOLDS['source'] == 'checked_training_split_quantiles'

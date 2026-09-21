"""Run in Colab: python -m pytest tests -q."""
import numpy as np
import pandas as pd
import pytest
from ml.preprocessing import FeatureCleaner, make_preprocessor, prepare_data, split_data
from ml.config import FEATURES


def row(**changes):
    value = dict(Age=21, Avg_Daily_Usage_Hours=4.0, Daily_Unlocks=130,
                 Study_Hours=4.5, Physical_Activity_Hours=2.2,
                 Sleep_Hours_Per_Night=6.7, Academic_Level='Undergraduate',
                 Most_Used_Platform='Facebook', Purpose_Of_Use='Networking',
                 Stress_Level='Medium', Mental_Health_Score=6.8)
    return value | changes


def test_cleaning_excludes_outcome_and_handles_impossible_values():
    frame = pd.DataFrame([row(Physical_Activity_Hours=-0.4)])
    result = FeatureCleaner().transform(frame)
    assert 'Mental_Health_Score' not in result
    assert 'Stress_Level' not in result
    assert np.isnan(result.loc[0, 'Physical_Activity_Hours'])
    assert frame.loc[0, 'Physical_Activity_Hours'] == -0.4


def test_imputation_statistics_do_not_change_on_transform():
    training = pd.DataFrame([row(Sleep_Hours_Per_Night=6), row(Sleep_Hours_Per_Night=8)])
    pipe = make_preprocessor()
    pipe.fit(training)
    imputer = pipe.named_steps['columns'].named_transformers_['numeric'].named_steps['impute']
    before = imputer.statistics_.copy()
    output = pipe.transform(pd.DataFrame([row(Sleep_Hours_Per_Night=100,
                                             Most_Used_Platform='Unseen platform')]))
    np.testing.assert_array_equal(before, imputer.statistics_)
    assert np.isfinite(output).all()


def test_duplicate_and_conflicting_profiles_are_removed_before_splitting():
    data = pd.DataFrame([row(), row(), row(Stress_Level='High'),
                         row(Age=22), row(Age=22, Mental_Health_Score=7)])
    cleaned, audit = prepare_data(data)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Age'] == 22
    assert audit['exact_duplicates_removed'] == 1
    assert audit['conflicting_profile_rows_removed'] == 2
    assert audit['repeated_profile_rows_removed'] == 1


def test_unknown_targets_fail_instead_of_silent_remapping():
    with pytest.raises(ValueError, match='Unknown stress labels'):
        prepare_data(pd.DataFrame([row(Stress_Level='Moderate')]))


def test_missing_feature_column_is_actionable():
    with pytest.raises(ValueError, match='Sleep_Hours_Per_Night'):
        prepare_data(pd.DataFrame([row()]).drop(columns=['Sleep_Hours_Per_Night']))


def test_splits_are_disjoint_and_preserve_four_classes():
    records = [row(Daily_Unlocks=70+i, Stress_Level=label)
               for i, label in enumerate(['Low', 'Medium', 'High', 'Very High'] * 30)]
    data, _ = prepare_data(pd.DataFrame(records))
    splits = split_data(data)
    sets = [set(splits[f'X_{part}'].index) for part in ('train', 'val', 'test')]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(set.union(*sets)) == 120
    for part in ('train', 'val', 'test'):
        assert set(splits[f'y_{part}']) == {'Low', 'Medium', 'High', 'Very High'}
        assert list(splits[f'X_{part}'].columns) == FEATURES

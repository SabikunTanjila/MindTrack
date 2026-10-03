"""Input-based lifestyle recommendations derived from training-data ranges.

Rules are transparent reflection prompts, not clinical thresholds. The fitted
classifier runs first; this module never changes its prediction.
"""
from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

import pandas as pd


# Fallbacks are rounded quantiles from the checked-in dataset's training split.
# Freshly exported artifacts replace them with values calculated from X_train.
DEFAULT_THRESHOLDS = {
    'source': 'checked_training_split_quantiles',
    'Sleep_Hours_Per_Night': {'low': 5.6, 'very_low': 5.0, 'high': 8.3, 'reference': 6.6},
    'Avg_Daily_Usage_Hours': {'high': 6.3, 'very_high': 7.3, 'reference': 5.0},
    'Daily_Unlocks': {'high': 204.0, 'very_high': 230.0, 'reference': 171.0},
    'Physical_Activity_Hours': {'low': 1.3, 'very_low': 0.9, 'reference': 1.7},
    'Study_Hours': {'high': 4.2, 'very_high': 5.3, 'reference': 2.8},
}

PRIORITY_ORDER = {'High': 0, 'Medium': 1, 'Low': 2}


def build_recommendation_thresholds(training_frame: pd.DataFrame) -> dict:
    """Calculate rule thresholds from the training partition only."""
    quantiles = training_frame[
        ['Sleep_Hours_Per_Night', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
         'Physical_Activity_Hours', 'Study_Hours']
    ].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    def value(feature: str, quantile: float) -> float:
        return round(float(quantiles.loc[quantile, feature]), 2)

    return {
        'source': 'training_split_quantiles',
        'Sleep_Hours_Per_Night': {
            'low': value('Sleep_Hours_Per_Night', 0.25),
            'very_low': value('Sleep_Hours_Per_Night', 0.10),
            'high': value('Sleep_Hours_Per_Night', 0.90),
            'reference': value('Sleep_Hours_Per_Night', 0.50),
        },
        'Avg_Daily_Usage_Hours': {
            'high': value('Avg_Daily_Usage_Hours', 0.75),
            'very_high': value('Avg_Daily_Usage_Hours', 0.90),
            'reference': value('Avg_Daily_Usage_Hours', 0.50),
        },
        'Daily_Unlocks': {
            'high': value('Daily_Unlocks', 0.75),
            'very_high': value('Daily_Unlocks', 0.90),
            'reference': value('Daily_Unlocks', 0.50),
        },
        'Physical_Activity_Hours': {
            'low': value('Physical_Activity_Hours', 0.25),
            'very_low': value('Physical_Activity_Hours', 0.10),
            'reference': value('Physical_Activity_Hours', 0.50),
        },
        'Study_Hours': {
            'high': value('Study_Hours', 0.75),
            'very_high': value('Study_Hours', 0.90),
            'reference': value('Study_Hours', 0.50),
        },
    }


def _number(values: Mapping[str, object], field: str) -> float | None:
    value = values.get(field)
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _text(values: Mapping[str, object], field: str, fallback: str) -> str:
    value = str(values.get(field, '')).strip()
    return value or fallback


def _recommendation(category: str, title: str, priority: str, reason: str,
                    action: str, feature: str | None, observed: float | None,
                    threshold: float | None, relevance: float) -> dict:
    return {
        'category': category,
        'title': title,
        'priority': priority,
        'reason': reason,
        # Kept for compatibility with the existing card renderer and clients.
        'description': reason,
        'action': action,
        'source_feature': feature,
        'observed_value': observed,
        'threshold_value': threshold,
        '_relevance': relevance,
    }


def _is_elevated(predicted_label: str, label_order: Sequence[str]) -> bool:
    labels = [str(label) for label in label_order]
    if predicted_label not in labels:
        return False
    return labels.index(predicted_label) >= (len(labels) + 1) // 2


def _priority_for_risk(predicted_label: str, label_order: Sequence[str]) -> str:
    """Keep action priority consistent with the model's ordered risk class."""
    labels = [str(label) for label in label_order]
    if predicted_label not in labels or len(labels) < 2:
        return 'Medium'
    position = labels.index(predicted_label)
    if position == 0:
        return 'Low'
    if position < (len(labels) + 1) // 2:
        return 'Medium'
    return 'High'


def generate_recommendations(
    values: Mapping[str, object],
    predicted_label: str,
    label_order: Sequence[str],
    thresholds: Mapping[str, object] | None = None,
    max_items: int = 6,
) -> dict:
    """Generate the five input-rule results plus optional wellbeing support."""
    rules = thresholds or DEFAULT_THRESHOLDS
    items: list[dict] = []

    sleep = _number(values, 'Sleep_Hours_Per_Night')
    screen = _number(values, 'Avg_Daily_Usage_Hours')
    unlocks = _number(values, 'Daily_Unlocks')
    activity = _number(values, 'Physical_Activity_Hours')
    study = _number(values, 'Study_Hours')
    platform = _text(values, 'Most_Used_Platform', 'your most-used platform')
    purpose = _text(values, 'Purpose_Of_Use', 'your usual purpose')
    priority = _priority_for_risk(predicted_label, label_order)

    sleep_rule = rules['Sleep_Hours_Per_Night']
    if sleep is not None and sleep < sleep_rule['low']:
        items.append(_recommendation(
            'Sleep', 'Make more room for sleep', priority,
            f'You reported {sleep:g} hours of sleep. A little more sleep may help you '
            'feel more rested and ready for the day.',
            'Try going to bed 15–30 minutes earlier for the next few nights.',
            'Sleep_Hours_Per_Night', sleep, sleep_rule['low'],
            (sleep_rule['low'] - sleep) / max(sleep_rule['low'], 1),
        ))
    elif sleep is not None and sleep > sleep_rule['high']:
        items.append(_recommendation(
            'Sleep', 'Check how rested you feel', priority,
            f'You reported {sleep:g} hours of sleep. Sleep length does not always show '
            'how restful your sleep was.',
            'Keep a regular sleep and wake time. If you often feel tired after sleeping, '
            'consider speaking with a qualified health professional.',
            'Sleep_Hours_Per_Night', sleep, sleep_rule['high'],
            (sleep - sleep_rule['high']) / max(sleep_rule['high'], 1),
        ))

    screen_rule = rules['Avg_Daily_Usage_Hours']
    if screen is not None and screen > screen_rule['high']:
        items.append(_recommendation(
            'Screen time', 'Take control of your screen time', priority,
            f'You reported {screen:g} hours of daily screen use, mainly using {platform} '
            f'for {purpose}. Long screen sessions can leave less time for rest and other activities.',
            f'Choose one {platform} session to shorten and set aside one screen-free period each day.',
            'Avg_Daily_Usage_Hours', screen, screen_rule['high'],
            (screen - screen_rule['high']) / max(screen_rule['high'], 1),
        ))

    unlock_rule = rules['Daily_Unlocks']
    if unlocks is not None and unlocks > unlock_rule['high']:
        items.append(_recommendation(
            'Digital interruptions', 'Reduce avoidable phone checks', priority,
            f'You reported checking your phone about {unlocks:g} times a day. Frequent '
            'checks can interrupt focus and relaxation.',
            'Turn off non-essential notifications and put your phone out of reach during '
            'one study, work, or rest period.',
            'Daily_Unlocks', unlocks, unlock_rule['high'],
            (unlocks - unlock_rule['high']) / max(unlock_rule['high'], 1),
        ))

    activity_rule = rules['Physical_Activity_Hours']
    if activity is not None and activity < activity_rule['low']:
        items.append(_recommendation(
            'Physical activity', 'Add a little more movement', priority,
            f'You reported {activity:g} hours of physical activity. A short movement '
            'break may help refresh your body and mind.',
            'Start with a short walk or stretch that fits your routine, then build up gradually.',
            'Physical_Activity_Hours', activity, activity_rule['low'],
            (activity_rule['low'] - activity) / max(activity_rule['low'], 1),
        ))

    study_rule = rules['Study_Hours']
    if study is not None and study > study_rule['high']:
        items.append(_recommendation(
            'Study balance', 'Plan breaks during study time', priority,
            f'You reported studying for {study:g} hours. Long study sessions can make it '
            'harder to rest, move, and stay focused.',
            'Study in shorter focused blocks, take a brief break between them, and choose '
            'a clear time to finish.',
            'Study_Hours', study, study_rule['high'],
            (study - study_rule['high']) / max(study_rule['high'], 1),
        ))

    if _is_elevated(predicted_label, label_order):
        age = _number(values, 'Age')
        action = (
            'If ongoing stress is affecting daily life, consider checking in with a '
            'trusted adult, school counselor, or qualified health professional.'
            if age is not None and age < 18 else
            'If ongoing stress is affecting daily life, consider checking in with '
            'someone you trust or a qualified counselor or health professional.'
        )
        items.append(_recommendation(
            'General wellbeing', 'Talk to someone if stress continues', priority,
            f'Your result shows a {predicted_label.lower()} stress level. This is a '
            'helpful signal to check in with yourself, but it is not a diagnosis.',
            action, None, None, None, 10.0,
        ))

    if not items:
        items.append(_recommendation(
            'Routine balance', 'Keep reviewing what works', priority,
            'Your answers did not highlight a specific habit that needs attention right now.',
            'Check in with your sleep, screen time, movement, and study balance each week. '
            'If something feels difficult, change one small thing at a time.',
            None, None, None, 0.0,
        ))

    ordered = sorted(
        items,
        key=lambda item: (PRIORITY_ORDER[item['priority']], -item['_relevance'], item['category']),
    )
    unique = []
    seen = set()
    for item in ordered:
        if item['category'] in seen:
            continue
        seen.add(item['category'])
        unique.append({key: value for key, value in item.items() if key != '_relevance'})
        if len(unique) >= max_items:
            break

    return {
        'predicted_risk': predicted_label,
        'threshold_source': rules.get('source', 'model_training_data'),
        'summary': (
            f'Your {predicted_label} result sets the priority shown below. The suggestions '
            'focus on the habits in your answers that may benefit from a small change.'
        ),
        'items': unique,
        'disclaimer': (
            'These are general lifestyle suggestions, not medical advice or a diagnosis.'
        ),
    }

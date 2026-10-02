"""Personalised lifestyle & stress-risk recommendation engine.

This module contains no clinical thresholds; benchmarks are dataset-informed
heuristics used for educational reflection only.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Lifestyle benchmarks (dataset-informed, not clinical guidelines)
# ---------------------------------------------------------------------------
_BENCHMARKS = {
    'Sleep_Hours_Per_Night':   {'ideal_min': 7.0, 'ideal_max': 8.0},
    'Avg_Daily_Usage_Hours':   {'warn_above': 5.0},
    'Daily_Unlocks':           {'warn_above': 85},
    'Physical_Activity_Hours': {'ideal_min': 0.5},
    'Study_Hours':             {'warn_above': 8.0},
}

# ---------------------------------------------------------------------------
# Cluster persona library
# ---------------------------------------------------------------------------
_CLUSTER_PERSONAS: dict[int, dict] = {
    0: {
        'name': 'Balanced Lifestyle',
        'insight': (
            'Your pattern resembles the Balanced cluster — moderate screen time, '
            'adequate sleep, and some regular physical activity. Maintaining this '
            'equilibrium is your biggest asset; protect it as demands increase.'
        ),
        'tips': [
            'Keep a consistent sleep–wake schedule even on weekends.',
            'Schedule short movement breaks to sustain your energy through the day.',
            'Periodically review your social-media usage so balance stays intentional.',
        ],
    },
    1: {
        'name': 'High Screen Time',
        'insight': (
            'Your pattern resembles the High Screen Time cluster — elevated daily '
            'social-media hours and frequent phone unlocks. Digital habits in this '
            'group are often linked with disrupted sleep and lower recovery time.'
        ),
        'tips': [
            'Set app-usage limits or a phone-free wind-down period before sleep.',
            'Replace one scroll session per day with a brief walk or stretch.',
            'Turn off non-essential notifications to reduce compulsive unlocks.',
        ],
    },
    2: {
        'name': 'Academic / Work Pressure',
        'insight': (
            'Your pattern resembles the Academic/Work Pressure cluster — long study '
            'or work hours that can crowd out rest and recovery activities. Sustained '
            'high effort without recovery often leads to diminishing returns.'
        ),
        'tips': [
            'Use the Pomodoro technique (25 min work / 5 min break) to build in rest.',
            'Block at least one non-study hour each day for physical activity or leisure.',
            'Protect your sleep window — it is where memory consolidation happens.',
        ],
    },
    3: {
        'name': 'Sedentary Pattern',
        'insight': (
            'Your pattern resembles the Sedentary cluster — low physical activity '
            'combined with extended sitting time. Even small increases in movement '
            'have been associated with improved mood and focus in student populations.'
        ),
        'tips': [
            'Aim for at least one 20-minute walk per day as a starting point.',
            'Stand up and stretch for 2 minutes every 45 minutes of sitting.',
            'Consider a light exercise routine three times per week to build consistency.',
        ],
    },
}

_DEFAULT_PERSONA = {
    'name': 'Mixed Profile',
    'insight': (
        'Your lifestyle pattern does not fit neatly into a single cluster profile. '
        'Review the individual metrics below for the most relevant reflection prompts.'
    ),
    'tips': [
        'Prioritise consistent sleep as a foundation for all other improvements.',
        'Build physical activity into your daily schedule in small, manageable blocks.',
        'Monitor your screen time and study hours to identify sources of overload.',
    ],
}

# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------
_RISK_PRIORITY_MAP: dict[str, dict[str, str]] = {
    'Very High': {'sleep': 'High',   'screen': 'High',   'unlocks': 'High',
                  'activity': 'High',   'study': 'High'},
    'High':      {'sleep': 'High',   'screen': 'High',   'unlocks': 'Medium',
                  'activity': 'High',   'study': 'Medium'},
    'Medium':    {'sleep': 'Medium', 'screen': 'Medium', 'unlocks': 'Low',
                  'activity': 'Medium', 'study': 'Medium'},
    'Low':       {'sleep': 'Low',    'screen': 'Low',    'unlocks': 'Low',
                  'activity': 'Low',    'study': 'Low'},
}

_PRIORITY_ORDER = {'High': 0, 'Medium': 1, 'Low': 2}


def _resolve_priority(feature_key: str, risk_level: str) -> str:
    mapping = _RISK_PRIORITY_MAP.get(risk_level, _RISK_PRIORITY_MAP['Low'])
    return mapping.get(feature_key, 'Low')


def _build_summary(risk_level: str, cluster_name: str, flagged_count: int) -> str:
    if risk_level == 'Very High':
        urgency = 'Your habits show several patterns strongly associated with high stress levels.'
    elif risk_level == 'High':
        urgency = 'Your habits show notable patterns linked to elevated stress.'
    elif risk_level == 'Medium':
        urgency = 'Your habits show some patterns that may contribute to moderate stress.'
    else:
        urgency = 'Your habits appear broadly balanced relative to the dataset.'

    if flagged_count == 0:
        action_note = 'No specific areas of concern were flagged — keep up the good work.'
    elif flagged_count == 1:
        action_note = 'One area has been flagged for your attention below.'
    else:
        action_note = f'{flagged_count} areas have been flagged for your attention below.'

    return (
        f'{urgency} You are placed in the "{cluster_name}" lifestyle cluster. '
        f'{action_note} These prompts are for reflection only and are not clinical advice.'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_personalized_recommendations(
    user_input: dict,
    predicted_risk: str,
    cluster_id: int,
) -> dict:
    """Generate educational, priority-ordered reflection prompts.

    Parameters
    ----------
    user_input:
        Raw assessment values keyed by dataset field names.
    predicted_risk:
        One of 'Low', 'Medium', 'High', 'Very High'.
    cluster_id:
        Integer K-Means cluster assignment.

    Returns
    -------
    dict with keys:
        risk_level, cluster_id, cluster_insight, overall_summary, actions
    """
    actions: list[dict] = []
    bench = _BENCHMARKS
    risk = predicted_risk if predicted_risk in _RISK_PRIORITY_MAP else 'Low'

    # Sleep ----------------------------------------------------------------
    sleep = user_input.get('Sleep_Hours_Per_Night')
    if sleep is not None:
        ideal_min = bench['Sleep_Hours_Per_Night']['ideal_min']
        ideal_max = bench['Sleep_Hours_Per_Night']['ideal_max']
        if sleep < ideal_min:
            deficit = round(ideal_min - sleep, 1)
            actions.append({
                'category': 'Sleep',
                'priority': _resolve_priority('sleep', risk),
                'title': 'Increase your nightly sleep',
                'description': (
                    f'You reported {sleep:.1f} hrs of sleep — {deficit} hr(s) below the '
                    f'{ideal_min}–{ideal_max} hr benchmark. Chronic short sleep is linked '
                    'to impaired concentration, mood regulation, and academic performance.'
                ),
                'action': (
                    f'Try moving your bedtime {int(deficit * 60)} minutes earlier this week '
                    'and keep a consistent wake time, including weekends.'
                ),
            })
        elif sleep > ideal_max + 1.5:
            actions.append({
                'category': 'Sleep',
                'priority': 'Low',
                'title': 'Check your sleep quality',
                'description': (
                    f'You reported {sleep:.1f} hrs of sleep, which is above the typical '
                    f'{ideal_max} hr benchmark. Excessive sleep time can sometimes signal '
                    'disrupted sleep quality rather than genuine extra rest.'
                ),
                'action': (
                    'Consider whether fatigue is driving extra sleep and evaluate your '
                    'sleep environment (light, temperature, device use).'
                ),
            })

    # Screen time ----------------------------------------------------------
    screen = user_input.get('Avg_Daily_Usage_Hours')
    if screen is not None:
        warn = bench['Avg_Daily_Usage_Hours']['warn_above']
        if screen > warn:
            excess = round(screen - warn, 1)
            actions.append({
                'category': 'Screen Time',
                'priority': _resolve_priority('screen', risk),
                'title': 'Reduce daily social-media usage',
                'description': (
                    f'You reported {screen:.1f} hrs/day of social-media use — '
                    f'{excess} hr(s) above the {warn} hr caution threshold. '
                    'High usage is correlated with disrupted sleep and increased '
                    'stress signals in student datasets.'
                ),
                'action': (
                    f'Set a daily cap of {max(1.0, warn):.0f} hrs using your device\'s '
                    'built-in screen-time tools and schedule two phone-free hours each evening.'
                ),
            })

    # Phone unlocks --------------------------------------------------------
    unlocks = user_input.get('Daily_Unlocks')
    if unlocks is not None:
        warn_u = bench['Daily_Unlocks']['warn_above']
        if unlocks > warn_u:
            actions.append({
                'category': 'Phone Habits',
                'priority': _resolve_priority('unlocks', risk),
                'title': 'Reduce compulsive phone checking',
                'description': (
                    f'You reported {unlocks} daily phone unlocks — above the '
                    f'{warn_u} unlock caution threshold. Frequent interruptions '
                    'fragment focus and make it harder to enter deep-work states.'
                ),
                'action': (
                    'Disable non-essential notifications and try leaving your phone '
                    'in another room during study blocks. Use "Do Not Disturb" during '
                    'sleep hours.'
                ),
            })

    # Physical activity ----------------------------------------------------
    activity = user_input.get('Physical_Activity_Hours')
    if activity is not None:
        ideal_act = bench['Physical_Activity_Hours']['ideal_min']
        if activity < ideal_act:
            actions.append({
                'category': 'Physical Activity',
                'priority': _resolve_priority('activity', risk),
                'title': 'Add movement to your daily routine',
                'description': (
                    f'You reported {activity:.1f} hrs of physical activity — below '
                    f'the {ideal_act} hr/day minimum benchmark. Regular movement supports '
                    'stress regulation, mood, and cognitive function.'
                ),
                'action': (
                    'Start with a 20-minute walk after lunch or dinner. Even light '
                    'activity three times per week produces measurable benefits in '
                    'student wellbeing studies.'
                ),
            })

    # Study load -----------------------------------------------------------
    study = user_input.get('Study_Hours')
    if study is not None:
        warn_s = bench['Study_Hours']['warn_above']
        if study > warn_s:
            actions.append({
                'category': 'Study Load',
                'priority': _resolve_priority('study', risk),
                'title': 'Build structured breaks into study sessions',
                'description': (
                    f'You reported {study:.1f} study hours — above the {warn_s} hr '
                    'sustained-study caution threshold. Extended unbroken study without '
                    'rest is associated with fatigue, poor retention, and burnout risk.'
                ),
                'action': (
                    'Use the Pomodoro technique: 25 minutes focused study followed by '
                    '5 minutes of rest. After four cycles take a 20–30 minute break. '
                    'Protect at least one full rest hour per day.'
                ),
            })

    # Sort primary actions: High → Medium → Low
    actions.sort(key=lambda a: (_PRIORITY_ORDER.get(a['priority'], 99), a['category']))
    flagged_count = len(actions)

    # Cluster persona & tips -----------------------------------------------
    persona = _CLUSTER_PERSONAS.get(cluster_id, _DEFAULT_PERSONA)
    for tip in persona['tips']:
        actions.append({
            'category': 'Lifestyle Tip',
            'priority': 'Low',
            'title': persona['name'],
            'description': tip,
            'action': tip,
        })

    # Re-sort with tips included
    actions.sort(key=lambda a: (_PRIORITY_ORDER.get(a['priority'], 99), a['category']))

    return {
        'risk_level': predicted_risk,
        'cluster_id': cluster_id,
        'cluster_insight': persona['insight'],
        'overall_summary': _build_summary(risk, persona['name'], flagged_count),
        'actions': actions,
    }


# ---------------------------------------------------------------------------
# Legacy shim — keeps predictor.py backward-compatible
# ---------------------------------------------------------------------------
def suggestions(values: dict, medians: dict) -> list[str]:
    """Deprecated: simple rule-based prompts kept for backward compatibility."""
    prompts: list[str] = []
    if values.get('Sleep_Hours_Per_Night', 99) < medians.get('Sleep_Hours_Per_Night', 7):
        prompts.append(
            'Your reported sleep is below the training-data median. '
            'Reflect on whether your schedule leaves enough time for rest.'
        )
    if values.get('Avg_Daily_Usage_Hours', 0) > medians.get('Avg_Daily_Usage_Hours', 3):
        prompts.append(
            'Your social-media time is above the training-data median. '
            'Consider which parts feel useful and where you might prefer a break.'
        )
    if values.get('Daily_Unlocks', 0) > medians.get('Daily_Unlocks', 60):
        prompts.append(
            'Your phone unlocks are above the training-data median. '
            'Consider whether notifications interrupt study or downtime.'
        )
    if not prompts:
        prompts.append(
            'Reflect on how your current routine supports rest, study, movement, and social connection.'
        )
    prompts.append(
        'These rule-based prompts use dataset comparisons; '
        'they are not treatment advice or explanations of the prediction.'
    )
    return prompts

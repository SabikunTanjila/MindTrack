"""Compatibility helpers for older MindTrack API clients."""
from ml.config import LABELS
from ml.recommendation import DEFAULT_THRESHOLDS, generate_recommendations


def generate_personalized_recommendations(user_input: dict, predicted_risk: str,
                                           cluster_id: int | None = None,
                                           thresholds: dict | None = None, **_) -> dict:
    """Return the legacy recommendation shape while using the new rule engine."""
    result = generate_recommendations(
        user_input, predicted_risk, LABELS, thresholds or DEFAULT_THRESHOLDS,
    )
    return {
        'risk_level': result['predicted_risk'],
        'cluster_id': cluster_id,
        'cluster_insight': result['summary'],
        'overall_summary': result['summary'],
        'actions': result['items'],
        'disclaimer': result['disclaimer'],
    }


def suggestions(values: dict, medians: dict) -> list[str]:
    """Small legacy prompt list retained in the prediction response."""
    prompts = []
    if values.get('Sleep_Hours_Per_Night', 99) < medians.get('Sleep_Hours_Per_Night', 7):
        prompts.append('A little more sleep may help you feel more rested.')
    if values.get('Avg_Daily_Usage_Hours', 0) > medians.get('Avg_Daily_Usage_Hours', 3):
        prompts.append('Consider setting aside one screen-free period each day.')
    if values.get('Daily_Unlocks', 0) > medians.get('Daily_Unlocks', 60):
        prompts.append('Turning off non-essential notifications may help reduce interruptions.')
    return prompts

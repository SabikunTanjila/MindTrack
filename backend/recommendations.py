"""Reflection prompts relative to this dataset; no medical thresholds."""


def suggestions(values, medians):
    prompts = []
    if values['Sleep_Hours_Per_Night'] < medians['Sleep_Hours_Per_Night']:
        prompts.append('Your reported sleep is below the training-data median. Reflect on whether your schedule leaves enough time for rest.')
    if values['Avg_Daily_Usage_Hours'] > medians['Avg_Daily_Usage_Hours']:
        prompts.append('Your social-media time is above the training-data median. Consider which parts feel useful and where you might prefer a break.')
    if values['Daily_Unlocks'] > medians['Daily_Unlocks']:
        prompts.append('Your phone unlocks are above the training-data median. Consider whether notifications interrupt study or downtime.')
    if not prompts:
        prompts.append('Reflect on how your current routine supports rest, study, movement, and social connection.')
    prompts.append('These rule-based prompts use dataset comparisons; they are not treatment advice or explanations of the prediction.')
    return prompts

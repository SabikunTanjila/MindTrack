# Dataset notes

File: `data/raw/mental_health_dataset.csv`.
SHA256: `32b542a497c39389735710fb4e2f43bdf444af5d9bacde6289801d201b6bebd3`.

The supplied CSV contains 5,000 rows, 13 columns, no blank fields, 2 exact
duplicate rows, and 10 negative Physical_Activity_Hours values (minimum -0.4).
Ages range from 18 to 24. Source URL, license, author, collection method, and
synthetic status cannot be verified from the CSV alone.

## Target (user-confirmed)

| Label | Raw count |
| --- | ---: |
| Low | 644 |
| Medium | 1,295 |
| High | 1,440 |
| Very High | 1,621 |

All four labels are preserved. These are dataset labels, not a clinically
validated stress scale. Post-cleaning counts are computed in Colab.

## Feature contract

| Column | Role | Interpretation / original range |
| --- | --- | --- |
| Age | Predictor | Years; 18–24 |
| Avg_Daily_Usage_Hours | Predictor / clustering | Daily social-media hours; 1–8.8 |
| Daily_Unlocks | Predictor / clustering | Daily phone unlocks; 62–273 |
| Study_Hours | Predictor / clustering | Hours; time window unverified; 0.3–8.3 |
| Physical_Activity_Hours | Predictor / clustering | Hours; time window unverified; -0.4–4.1 |
| Sleep_Hours_Per_Night | Predictor / clustering | Hours per night; 3.6–9.9 |
| Academic_Level | Predictor | High School, Undergraduate, Graduate |
| Most_Used_Platform | Predictor | 12 platform categories |
| Purpose_Of_Use | Predictor | Education, Entertainment, Networking, News |
| Gender | Excluded | Not needed for the lifestyle assessment |
| Country | Excluded | 111 categories; omitted from the questionnaire/model |
| Mental_Health_Score | Excluded | Outcome-like score; possible target leakage |
| Stress_Level | Target only | Four original labels |

## Cleaning and splitting

1. Remove exact duplicates and report the count.
2. Trim strings, validate target labels, and drop missing targets without imputation.
3. Convert invalid/nonfinite/impossible numeric values to missing values.
4. Remove ambiguous predictor profiles with conflicting labels; retain one copy
   of repeated profiles with the same target. Report both removal counts.
5. Stratify unique profiles into 60% training, 20% validation, 20% test (seed 42).
6. Fit imputation, scaling, and encoding only in training pipelines and CV folds.

Negative activity becomes missing rather than being guessed as zero. A broad
0–24 hour sanity bound is used for this file, not as a health recommendation.
Confirm study/activity time windows with the original source. No participant ID
is present, so subject overlap cannot be ruled out. Global importance does not
reveal what caused an individual's stress.

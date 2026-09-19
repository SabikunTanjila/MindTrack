# MindTrack: lifestyle clustering and stress-label classification

Generated from the run at 2026-09-09T07:59:51.936397+00:00. Numeric results below are
measured outputs for the dataset used in this run.

## Objective

Compare KNN, Logistic Regression, and Random Forest for predicting the dataset's
Low, Medium, High, and Very High stress labels. Use K-Means independently to
identify lifestyle profiles and demonstrate the workflow through a web dashboard.

## Dataset and preparation

The input contained 5000 rows; 4993 unique,
unambiguous predictor profiles remained after cleaning. The audit removed
2 exact duplicates,
0 conflicting-profile rows, and
5 additional repeated-profile rows.
Missing targets removed: 0.

Predictors: Age, Avg_Daily_Usage_Hours, Daily_Unlocks, Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night, Academic_Level, Most_Used_Platform, Purpose_Of_Use.
The target and Mental_Health_Score are excluded from predictor inputs. Gender
and country are omitted. Invalid numeric values become missing; training-fitted
median/mode imputers handle missing values. Scaling and one-hot encoding occur
inside cross-validation pipelines. See metrics/data_audit.json for all counts.

## Experimental method

Stratified partitions contain 2995 training, 999 validation,
and 999 test records (seed 42). Classifier parameters
are tuned with three-fold stratified CV on the training partition. The primary
model, **Random Forest**, is selected by validation macro F1 before test evaluation.
A majority-class classifier provides a baseline. No model is selected from the
test results below. Group/participant IDs are unavailable, so subject overlap
cannot be ruled out even after removing identical predictor profiles.

## Held-out classification results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | OVR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN | 0.8438 | 0.8574 | 0.8328 | 0.8431 | 0.9522 |
| Logistic Regression | 0.7397 | 0.7362 | 0.7256 | 0.7303 | 0.9258 |
| Random Forest | 0.8679 | 0.8752 | 0.8652 | 0.8697 | 0.9774 |
| Majority baseline | 0.3243 | 0.0811 | 0.2500 | 0.1224 | 0.5000 |

See metrics/classification.json for per-class precision/recall/F1 and confusion
matrix counts. figures/confusion_matrices.png visualizes the classification
errors. Macro F1 weights each class equally despite unequal class frequencies.

## Unsupervised lifestyle analysis

K-Means uses five numeric behavior inputs from training data without target
labels. The selected cluster count is **2**.

| k | Silhouette | Inertia |
| ---: | ---: | ---: |
| 2 | 0.4631 | 6393.68 |
| 3 | 0.3457 | 4692.96 |
| 4 | 0.2828 | 3944.36 |
| 5 | 0.2729 | 3528.70 |
| 6 | 0.2537 | 3163.42 |

Cluster names describe the strongest deviations from standardized training
averages. Cluster IDs are not risk levels. PCA is a two-dimensional visualization,
not the feature space used to fit K-Means. See figures/clustering.png and
metrics/cluster_profiles.csv for the observed profiles.

## Explainability and application

Global permutation importance measures the drop in validation macro F1 after
shuffling an input. Correlated features may share or mask importance. The
dashboard shows these global values separately from input-versus-median
comparisons, which do not establish individual causes. Reflection prompts are
rule-based and are not medical recommendations.

FastAPI serves validated predictions and the same-origin dashboard inside a
hosted Colab runtime. Saved pipelines preserve the transformations used during
training. The notebook performs behavior tests and a real HTTP smoke check;
consult metrics/verification.txt for the observed test outcome from this run.

## Limitations

- Educational estimates of dataset labels, not a clinical assessment.
- Dataset source, license, collection method, and synthetic status are unverified.
- The dataset covers ages 18–24; performance outside this cohort is untested.
- Probabilities are uncalibrated model outputs.
- Study and physical-activity time windows need source confirmation.
- Global feature importance and input comparisons do not establish individual causes.

## Conclusion

In this run, Random Forest achieved test macro F1 **0.8697** and
accuracy **0.8679**. These values describe performance on this
dataset partition. They do not establish generalization to a new population or
clinical usefulness. Source documentation and an independent real-world dataset
would be needed before stronger claims.

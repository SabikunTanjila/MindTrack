# MindTrack: Colab implementation design

The existing project plan is implemented for a hosted Google Colab CPU runtime.
The user's local workspace is used only to author and package files. No local
training, dependency installation, server, or application execution is required.

## Dataset and target

Use only `data/raw/mental_health_dataset.csv`. Preserve the four user-approved
labels: Low, Medium, High, Very High. The file contains 5,000 rows, 13 columns,
2 exact duplicate rows, and 10 negative physical-activity entries. Its source,
license, collection process, and synthetic status are not established by the CSV.

Use age, daily social-media usage, daily unlocks, study hours, physical activity,
sleep hours, academic level, platform, and purpose as predictors. Exclude the
target and Mental_Health_Score from all predictors. Exclude gender and country
from the questionnaire/model. Restrict assessment ages to the observed 18–24 cohort.
Study and activity time units require source confirmation; do not make threshold
health advice from these fields.

## Machine learning

Remove exact duplicates. Normalize input strings and convert impossible numeric
values to missing values using fixed domain bounds. Drop repeated predictor
profiles before splitting; discard ambiguous profiles with conflicting targets.
Report every removal. Never impute target labels.

Use stratified 60/20/20 train/validation/test splits with seed 42. Fit imputers,
encoders, and scalers inside each training/CV pipeline. Tune KNN, Logistic
Regression, and Random Forest with three-fold stratified CV; select the primary
classifier by validation macro F1. Freeze it before test evaluation. Include a
majority-class baseline. Report accuracy, macro precision/recall/F1, weighted F1,
balanced accuracy, multiclass OVR AUC, and confusion matrices.

Fit K-Means separately on five numeric behavior fields from the training split.
Compare k=2..6 using inertia and sampled silhouette; choose by silhouette. Fit
PCA on training features for visualization. Describe clusters using observed
profiles, without assigning stress levels to cluster IDs.

Compute global permutation importance on validation data. For an individual,
show inputs relative to training medians; do not call this a causal explanation.
Probabilities are uncalibrated model estimates, not clinical confidence.

## Colab and application

Deliver a self-contained `MindTrack_Colab.ipynb` with embedded source files and
guided cells for setup, upload, audit, EDA, preparation, clustering, classification,
evaluation/export, verification, API, and dashboard. Also populate the four topic
notebooks for group work. Each topic notebook includes its own prerequisite cells.

The dashboard uses HTML/CSS/JavaScript served by FastAPI in Colab and calls the
real `/predict` endpoint. Google Colab's `serve_kernel_port_as_iframe` embeds it.
No public tunnel, external frontend build, or local runtime is needed. Shutdown
and downloadable result archives are notebook actions. Runtime files disappear
on reset unless downloaded; source can be restored by rerunning setup.

## Verification

Ship behavior tests for leakage prevention, malformed inputs, split isolation,
serialization, probability labels, classifier metrics, and API failure behavior.
Run them in Colab, including a small synthetic integration fixture. Do not claim
Colab execution or measured model scores before actually observing them. Here,
verify notebook JSON, source embedding, paths, and packaging without running the
project on the local computer.

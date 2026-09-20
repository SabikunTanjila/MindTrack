# Project plan — Colab edition

The original three-week plan is adapted to a hosted Colab runtime and the
user-approved four-label dataset.

## Stage 1: Data and ML

Audit, clean impossible values, remove repeated/conflicting predictor profiles,
split 60/20/20, explore training data, and construct preprocessing pipelines.
Train K-Means separately from the classifiers. Tune classifiers with CV, select
by validation macro F1, and evaluate on the held-out test set.

## Stage 2: Backend and dashboard

Serialize complete fitted pipelines and expose them through FastAPI. Serve a
responsive HTML/CSS/JavaScript assessment and results dashboard inside Colab.
The optional React frontend is replaced by a build-free frontend so the complete
project runs in the requested hosted notebook environment.

## Stage 3: Explanation, verification, delivery

Show global importance, input comparisons, and reflection prompts. Run behavior
tests, check the actual HTTP API, and display the dashboard through Colab's iframe
helper. Download models, figures, metrics, and environment versions. Use actual
generated outputs for the report and presentation.

Use MindTrack_Colab.ipynb to follow all stages, or divide the topic notebooks among
group members. Authentication, databases, history, and SHAP remain optional.

See [implementation plan](colab-implementation-plan.md), [design](colab-design.md),
and [Colab guide](colab-guide.md).

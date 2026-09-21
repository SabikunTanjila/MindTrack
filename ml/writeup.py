"""Generate a lab report and presentation notes from measured run outputs."""
from pathlib import Path


def generate_writeup(metadata, evaluation, output_dir):
    report_dir = Path(output_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    primary = metadata['primary_model']
    selected = evaluation[primary]
    audit = metadata['dataset_audit']
    split = metadata['split_sizes']
    table = ['| Model | Accuracy | Macro precision | Macro recall | Macro F1 | OVR AUC |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, metrics in evaluation.items():
        auc = metrics['roc_auc_ovr_macro']
        values = [name] + [f'{metrics[key]:.4f}' for key in
                          ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
        values.append('N/A' if auc is None else f'{auc:.4f}')
        table.append('| ' + ' | '.join(values) + ' |')
    cluster_table = ['| k | Silhouette | Inertia |', '| ---: | ---: | ---: |']
    for row in metadata['cluster_scores']:
        cluster_table.append(f'| {row["k"]} | {row["silhouette"]:.4f} | {row["inertia"]:.2f} |')
    features = ', '.join(metadata['features'])
    limitations = '\n'.join('- ' + line for line in metadata['limitations'])
    report = f'''# MindTrack: lifestyle clustering and stress-label classification

Generated from the run at {metadata['created_utc']}. Numeric results below are
measured outputs for the dataset used in this run.

## Objective

Compare KNN, Logistic Regression, and Random Forest for predicting the dataset's
Low, Medium, High, and Very High stress labels. Use K-Means independently to
identify lifestyle profiles and demonstrate the workflow through a web dashboard.

## Dataset and preparation

The input contained {audit['raw_rows']} rows; {audit['prepared_rows']} unique,
unambiguous predictor profiles remained after cleaning. The audit removed
{audit['exact_duplicates_removed']} exact duplicates,
{audit['conflicting_profile_rows_removed']} conflicting-profile rows, and
{audit['repeated_profile_rows_removed']} additional repeated-profile rows.
Missing targets removed: {audit['missing_target_rows_removed']}.

Predictors: {features}.
The target and Mental_Health_Score are excluded from predictor inputs. Gender
and country are omitted. Invalid numeric values become missing; training-fitted
median/mode imputers handle missing values. Scaling and one-hot encoding occur
inside cross-validation pipelines. See metrics/data_audit.json for all counts.

## Experimental method

Stratified partitions contain {split['train']} training, {split['val']} validation,
and {split['test']} test records (seed {metadata['seed']}). Classifier parameters
are tuned with three-fold stratified CV on the training partition. The primary
model, **{primary}**, is selected by validation macro F1 before test evaluation.
A majority-class classifier provides a baseline. No model is selected from the
test results below. Group/participant IDs are unavailable, so subject overlap
cannot be ruled out even after removing identical predictor profiles.

## Held-out classification results

{chr(10).join(table)}

See metrics/classification.json for per-class precision/recall/F1 and confusion
matrix counts. figures/confusion_matrices.png visualizes the classification
errors. Macro F1 weights each class equally despite unequal class frequencies.

## Unsupervised lifestyle analysis

K-Means uses five numeric behavior inputs from training data without target
labels. The selected cluster count is **{metadata['cluster_count']}**.

{chr(10).join(cluster_table)}

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

{limitations}

## Conclusion

In this run, {primary} achieved test macro F1 **{selected['f1_macro']:.4f}** and
accuracy **{selected['accuracy']:.4f}**. These values describe performance on this
dataset partition. They do not establish generalization to a new population or
clinical usefulness. Source documentation and an independent real-world dataset
would be needed before stronger claims.
'''
    (report_dir / 'project_report.md').write_text(report, encoding='utf-8')
    slides = f'''# MindTrack presentation notes

1. **Problem and scope:** lifestyle patterns and four dataset stress labels;
   educational use, not diagnosis.
2. **Dataset:** {audit['raw_rows']} input rows, {audit['prepared_rows']} usable
   profiles; show the quality audit and explain unverified provenance.
3. **Preprocessing:** deduplication, invalid values, split isolation, imputation,
   scaling, and one-hot encoding; explain exclusion of Mental_Health_Score.
4. **Clustering:** show figures/clustering.png; explain why the selected
   k={metadata['cluster_count']} does not need to equal the number of classes.
5. **Classification:** KNN/LR/RF, majority baseline, three-fold CV, validation
   macro F1 selection, then held-out evaluation.
6. **Results:** show metrics/classification.csv and confusion matrices.
   Primary: {primary}; measured test macro F1: {selected['f1_macro']:.4f}.
7. **Demo:** run a valid assessment, compare model predictions, then show the
   research tab. Distinguish probabilities, global importance, and comparisons.
8. **Limitations and next steps:** source/units confirmation, calibration,
   independent validation, and reproducibility using saved runtime versions.

Use actual notebook outputs for screenshots. State clearly if presenting a
synthetic test fixture rather than the supplied project dataset.
'''
    (report_dir / 'presentation_notes.md').write_text(slides, encoding='utf-8')

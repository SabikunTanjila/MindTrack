# File-authoring utility only: creates notebooks without executing project code.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectUtf8 = New-Object System.Text.UTF8Encoding($false)
$projectEmbeddedFiles = [ordered]@{}
foreach ($projectFolder in @('ml', 'backend', 'frontend', 'tests')) {
    Get-ChildItem -LiteralPath (Join-Path $projectRoot $projectFolder) -File -Recurse |
        Where-Object { $_.Extension -in @('.py', '.html', '.css', '.js') } |
        Sort-Object FullName | ForEach-Object {
            $projectRelative = $_.FullName.Substring($projectRoot.Length + 1).Replace('\', '/')
            $projectEmbeddedFiles[$projectRelative] = [System.IO.File]::ReadAllText($_.FullName)
        }
}
foreach ($projectFile in @('requirements.txt', 'README.md', 'docs/dataset-notes.md', 'docs/colab-guide.md', 'docs/verification.md', 'docs/project-plan.md', 'docs/colab-design.md')) {
    $projectEmbeddedFiles[$projectFile] = [System.IO.File]::ReadAllText((Join-Path $projectRoot $projectFile))
}
$projectJson = $projectEmbeddedFiles | ConvertTo-Json -Depth 10 -Compress
$projectPayload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($projectJson))

$projectGuard = @'
# Step 1 — Confirm that execution is in a hosted Colab runtime.
from pathlib import Path
import os, sys
if not Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'):
    raise RuntimeError('Open this notebook in Google Colab and connect to a hosted Python 3 runtime. Do not use a local runtime.')
ROOT = Path('/content/mindtrack')
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('Hosted Colab workspace:', ROOT)
'@
$projectBootstrap = @"
# Step 2 — Restore the project source embedded in this notebook.
# No GitHub repository, project ZIP, or local Python environment is needed.
import base64, json
SOURCE_FILES = json.loads(base64.b64decode('$projectPayload'))
for relative, content in SOURCE_FILES.items():
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
for folder in ['data/raw', 'data/processed', 'models', 'reports/figures', 'reports/metrics']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
print(f'Restored {len(SOURCE_FILES)} source files.')
"@
$projectInstall = @'
# Step 3 — Install dependencies in the Colab VM, never on your computer.
# Run this before importing pandas, numpy, sklearn, or backend modules.
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
print('Dependency installation finished. Continue with the upload cell.')
'@
$projectUpload = @'
# Step 4 — Upload the supplied mental_health_dataset.csv when prompted.
from google.colab import files
import hashlib
DATASET = ROOT / 'data/raw/mental_health_dataset.csv'
if not DATASET.exists():
    uploaded = files.upload()
    candidates = [name for name in uploaded if Path(name).name == 'mental_health_dataset.csv']
    if len(candidates) != 1:
        raise ValueError('Select mental_health_dataset.csv from your project data/raw folder, then rerun this cell.')
    DATASET.write_bytes(uploaded[candidates[0]])
    del uploaded
digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
print('Dataset:', DATASET.name, '| bytes:', DATASET.stat().st_size)
print('SHA256:', digest)
if digest != '32b542a497c39389735710fb4e2f43bdf444af5d9bacde6289801d201b6bebd3':
    print('NOTE: This file differs from the originally inspected CSV. Review its audit before training.')
'@
$projectPrepare = @'
# Step 5 — Audit, clean with fixed rules, and split before statistical EDA.
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ml.config import FEATURES, LABELS, NUMERIC, TARGET, SEED
from ml.preprocessing import prepare_data, split_data, make_preprocessor
from ml.reporting import save_json
raw = pd.read_csv(DATASET)
prepared, audit = prepare_data(raw)
splits = split_data(prepared)
print(json.dumps(audit, indent=2))
display(pd.DataFrame({part: splits[f'y_{part}'].value_counts().reindex(LABELS)
                      for part in ['train','val','test']}))
save_json(ROOT/'reports/metrics/data_audit.json', audit)
prepared.to_csv(ROOT/'data/processed/model_dataset.csv', index=False)
assignments = pd.concat([pd.DataFrame({'prepared_row': splits[f'X_{part}'].index, 'split': part})
                         for part in ['train','val','test']], ignore_index=True)
assignments.to_csv(ROOT/'reports/metrics/split_assignments.csv', index=False)
assert 'Mental_Health_Score' not in splits['X_train'].columns
print('Splits:', {part: len(splits[f'X_{part}']) for part in ['train','val','test']})
'@
$projectEDA = @'
# Step 6 — Explore only the training partition to preserve the holdout.
from ml.reporting import plot_eda
training_frame = splits['X_train'].join(splits['y_train'])
display(training_frame.head())
display(training_frame[NUMERIC].describe().T)
for figure in plot_eda(training_frame, ROOT/'reports/figures'):
    display(figure)
    plt.close(figure)
'@
$projectPreprocessing = @'
# Step 7 — Inspect preprocessing without fitting on validation/test inputs.
preview_preprocessor = make_preprocessor()
train_matrix = preview_preprocessor.fit_transform(splits['X_train'])
val_matrix = preview_preprocessor.transform(splits['X_val'])
test_matrix = preview_preprocessor.transform(splits['X_test'])
print('Transformed shapes:', train_matrix.shape, val_matrix.shape, test_matrix.shape)
print('Encoded features:', preview_preprocessor.get_feature_names_out().tolist())
import numpy as np
assert np.isfinite(train_matrix).all()
assert np.isfinite(val_matrix).all()
assert np.isfinite(test_matrix).all()
print('Training-only preprocessing inspection passed.')
# Training below creates independent pipelines inside each CV fold.
'@
$projectClusters = @'
# Step 8 — Discover lifestyle profiles independently of stress labels.
from ml.clustering import train_clusters
from ml.reporting import plot_clusters
clustering = train_clusters(splits['X_train'])
display(clustering['scores'])
display(clustering['profiles'])
print('Selected cluster count:', clustering['best_k'])
for cluster_id, description in clustering['descriptions'].items():
    print(f'Cluster {int(cluster_id)+1}: {description}')
figure = plot_clusters(clustering, ROOT/'reports/figures')
display(figure)
plt.close(figure)
clustering['scores'].to_csv(ROOT/'reports/metrics/clustering.csv', index=False)
clustering['profiles'].to_csv(ROOT/'reports/metrics/cluster_profiles.csv')
'@
$projectTrain = @'
# Step 9 — Tune classifiers with three-fold CV on training data.
# CPU is sufficient. This is the longest step; wait for each model's result.
from ml.train import train_classifiers
training = train_classifiers(splits)
display(training['selection'])
print('Primary model selected before test evaluation:', training['primary_model'])
'@
$projectEvaluate = @'
# Step 10 — Evaluate once on the held-out test split and explain globally.
from ml.evaluate import evaluate_classifiers, global_importance
from ml.reporting import metric_table, plot_evaluation
evaluation = evaluate_classifiers(training, splits)
importance = global_importance(training, splits)
display(metric_table(evaluation).round(4))
display(importance)
for figure in plot_evaluation(evaluation, importance, ROOT/'reports/figures'):
    display(figure)
    plt.close(figure)
print('Do not choose or retune models using these test scores.')
print('Dataset provenance is unverified; high scores do not establish clinical usefulness.')
'@
$projectExport = @'
# Step 11 — Save complete fitted pipelines, metadata, and real evaluation results.
from ml.reporting import export_artifacts
artifact = export_artifacts(training, clustering, evaluation, importance, splits,
                            audit, ROOT, dataset_path=DATASET)
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True)
(ROOT/'reports/metrics/runtime_packages.txt').write_text(freeze.stdout, encoding='utf-8')
print('Saved:', artifact)
print('Only load joblib files that you created or trust.')
'@
$projectTests = @'
# Step 12 — Run behavior tests in Colab, including synthetic training/API checks.
# Synthetic fixture results are tests, not performance claims about your dataset.
test_run = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'],
                          capture_output=True, text=True)
log = test_run.stdout + test_run.stderr
print(log)
(ROOT/'reports/metrics/verification.txt').write_text(log, encoding='utf-8')
if test_run.returncode != 0:
    raise RuntimeError('Verification failed. Read the failure above before starting the demo.')
'@
$projectServer = @'
# Step 13 — Start the real FastAPI server inside the Colab VM.
import threading, time, urllib.request, urllib.error
import uvicorn
from backend.main import create_app
if 'mindtrack_server' in globals():
    mindtrack_server.should_exit = True
    mindtrack_thread.join(timeout=5)
    if mindtrack_thread.is_alive():
        raise RuntimeError('Previous server did not stop. Restart the Colab session and rerun.')
app = create_app(ROOT/'models')
mindtrack_server = uvicorn.Server(uvicorn.Config(app, host='0.0.0.0', port=8000, log_level='warning'))
mindtrack_thread = threading.Thread(target=mindtrack_server.run, daemon=True)
mindtrack_thread.start()
for attempt in range(100):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
            health = json.load(response)
        break
    except (OSError, urllib.error.URLError):
        time.sleep(.1)
else:
    raise RuntimeError('Colab API did not start; inspect the server error above.')
assert health['models_loaded'], health
sample = splits['X_train'].dropna().iloc[0].to_dict()
sample['Age'], sample['Daily_Unlocks'] = int(sample['Age']), int(sample['Daily_Unlocks'])
request = urllib.request.Request('http://127.0.0.1:8000/predict',
    data=json.dumps(sample).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request) as response:
    smoke_result = json.load(response)
assert set(smoke_result['probabilities']) == set(LABELS)
assert abs(sum(smoke_result['probabilities'].values())-1) < 1e-6
invalid_request = urllib.request.Request('http://127.0.0.1:8000/predict',
    data=json.dumps(sample | {'Sleep_Hours_Per_Night': -1}).encode(),
    headers={'Content-Type':'application/json'})
try:
    urllib.request.urlopen(invalid_request)
    raise AssertionError('The API accepted an invalid negative sleep value.')
except urllib.error.HTTPError as error:
    assert error.code == 422, error.code
save_json(ROOT/'reports/metrics/api_smoke.json',
          {'health': health, 'valid_prediction': 'passed', 'invalid_input_422': 'passed'})
print('Real HTTP API check passed:', health, '| sample label:', smoke_result['label'])
'@
$projectDashboard = @'
# Step 14 — Open the dashboard through Colab's kernel-port iframe.
# The app runs in the Colab VM; your browser displays it here.
from google.colab import output
output.serve_kernel_port_as_iframe(8000, path='/', height='1100')
'@
$projectDownload = @'
# Step 15 — Download models, reports, and docs before the runtime resets.
# The results archive excludes the raw dataset and individual assessment inputs.
from zipfile import ZipFile, ZIP_DEFLATED
archive = Path('/content/MindTrack_results.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
    for folder in ['models','reports','docs']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.name != '.gitkeep':
                bundle.write(path, path.relative_to(ROOT))
    bundle.write(ROOT/'requirements.txt', 'requirements.txt')
files.download(str(archive))
'@
$projectShutdown = @'
# Optional — Set True only when you are finished using the dashboard.
STOP_DEMO = False
if STOP_DEMO and 'mindtrack_server' in globals():
    mindtrack_server.should_exit = True
    mindtrack_thread.join(timeout=5)
    print('Colab API stopped.')
'@

function New-ProjectMarkdown([string]$Text) {
    return @{cell_type='markdown'; metadata=@{}; source=@($Text)}
}
function New-ProjectCode([string]$Text) {
    return @{cell_type='code'; metadata=@{}; execution_count=$null; outputs=@(); source=@($Text)}
}
function Write-ProjectNotebook([string]$RelativePath, [string]$Title, [string[]]$Stages) {
    $projectCells = New-Object System.Collections.Generic.List[object]
    $projectCells.Add((New-ProjectMarkdown "# $Title`n`n**MindTrack · AI Lab · Google Colab CPU runtime**`n`nRun cells from top to bottom. This notebook embeds the project code; upload only your supplied mental_health_dataset.csv when asked. Use a hosted Python 3 runtime, not a local runtime. The four labels are **Low, Medium, High, Very High**.`n`nThe CSV has 5,000 rows, 2 duplicate rows, and 10 negative activity values in the original audit. Its provenance is unverified. This is an educational estimate of dataset stress labels, not a clinical assessment.`n`nA runtime reset removes files and stops the demo. Download your results before leaving. No GPU, API key, Node installation, or public tunnel is required."))
    $projectCells.Add((New-ProjectCode $projectGuard))
    $projectCells.Add((New-ProjectCode $projectBootstrap))
    $projectCells.Add((New-ProjectCode $projectInstall))
    $projectCells.Add((New-ProjectMarkdown "## Dataset upload`nChoose the file from **data/raw/mental_health_dataset.csv** on your computer. Colab cannot access that local path automatically. Rerunning the upload cell reuses the runtime copy. To switch files, remove that copy in Colab's Files panel first."))
    $projectCells.Add((New-ProjectCode $projectUpload))
    $projectCells.Add((New-ProjectMarkdown "## Audit and split`nFixed rules convert impossible values to missing values. Exact duplicates, conflicting labeled profiles, and repeated profiles are reported and removed before splitting. Imputation is learned later, inside training pipelines. The original four target labels are preserved."))
    $projectCells.Add((New-ProjectCode $projectPrepare))
    foreach ($projectStage in $Stages) {
        switch ($projectStage) {
            'eda' { $projectCells.Add((New-ProjectMarkdown "## Exploratory data analysis`nInspect feature distributions, relationships, and possible outliers on training data. Plausible extremes are retained; test data is reserved for final evaluation.")); $projectCells.Add((New-ProjectCode $projectEDA)) }
            'preprocess' { $projectCells.Add((New-ProjectMarkdown "## Preprocessing`nNumeric features use median imputation and standardization. Categories use most-frequent imputation and one-hot encoding that tolerates unseen categories. Gender, Country, Stress_Level, and Mental_Health_Score are excluded from predictor inputs.")); $projectCells.Add((New-ProjectCode $projectPreprocessing)) }
            'clusters' { $projectCells.Add((New-ProjectMarkdown "## K-Means clustering`nCompare k=2..6 using silhouette and inertia. PCA gives a two-dimensional view. Clusters describe lifestyle patterns and are not stress classes; k does not have to equal four.")); $projectCells.Add((New-ProjectCode $projectClusters)) }
            'train' { $projectCells.Add((New-ProjectMarkdown "## Classifier training and selection`nTune KNN, Logistic Regression, and Random Forest with three-fold stratified CV. Select the primary classifier by validation macro F1. A majority-class baseline makes the performance comparison meaningful.")); $projectCells.Add((New-ProjectCode $projectTrain)) }
            'evaluate' { $projectCells.Add((New-ProjectMarkdown "## Final evaluation and global importance`nReport test accuracy, macro precision/recall/F1, weighted F1, balanced accuracy, multiclass OVR AUC, and confusion matrices. Global importance uses validation data. These values are generated by execution; no scores are prefilled.")); $projectCells.Add((New-ProjectCode $projectEvaluate)); $projectCells.Add((New-ProjectCode $projectExport)) }
            'tests' { $projectCells.Add((New-ProjectMarkdown "## Verification`nThese tests exercise data leakage safeguards, invalid inputs, split isolation, serialization, and real API routes. A small synthetic fixture keeps test training inexpensive. It does not measure the supplied dataset's performance.")); $projectCells.Add((New-ProjectCode $projectTests)) }
            'demo' { $projectCells.Add((New-ProjectMarkdown "## API and interactive dashboard`nStart the API in the Colab VM, check it with a real HTTP request, and display the dashboard in an iframe. If the iframe is blank, check your browser's permissions for Colab embedded content and rerun the display cell. The notebook's result tables remain available even if the browser blocks the iframe.")); $projectCells.Add((New-ProjectCode $projectServer)); $projectCells.Add((New-ProjectCode $projectDashboard)) }
            'download' { $projectCells.Add((New-ProjectMarkdown "## Keep your work`nDownload the generated results archive before disconnecting. It contains your trained artifacts and reports. The separate local project folder remains your editable source.")); $projectCells.Add((New-ProjectCode $projectDownload)) }
            'shutdown' { $projectCells.Add((New-ProjectCode $projectShutdown)) }
        }
    }
    $projectCells.Add((New-ProjectMarkdown "## References and interpretation`n- [Google Colab file I/O](https://colab.research.google.com/notebooks/io.ipynb)`n- [Google's kernel-port iframe implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py)`n- [Scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)`n`nA high holdout score can reflect patterns or label-generation rules within this dataset. It is not evidence of clinical validity or generalization to another population. Confirm the dataset's origin and the units for Study_Hours and Physical_Activity_Hours before making substantive interpretations."))
    for ($projectCellIndex=0; $projectCellIndex -lt $projectCells.Count; $projectCellIndex++) {
        $projectCells[$projectCellIndex]['id'] = 'cell-' + $projectCellIndex.ToString('000')
    }
    $projectNotebook = @{cells=@($projectCells.ToArray()); metadata=@{colab=@{name=(Split-Path -Leaf $RelativePath); provenance=@()}; kernelspec=@{display_name='Python 3';language='python';name='python3'};language_info=@{name='python'}};nbformat=4;nbformat_minor=5}
    $projectDestination = Join-Path $projectRoot $RelativePath
    [IO.Directory]::CreateDirectory((Split-Path -Parent $projectDestination)) | Out-Null
    [IO.File]::WriteAllText($projectDestination, ($projectNotebook | ConvertTo-Json -Depth 30), $projectUtf8)
    Write-Output "$RelativePath : $($projectCells.Count) cells"
}
Write-ProjectNotebook 'MindTrack_Colab.ipynb' 'MindTrack — Complete Colab Project' @('eda','preprocess','clusters','train','evaluate','tests','demo','download','shutdown')
Write-ProjectNotebook 'notebooks/01_eda.ipynb' 'MindTrack — 01 · Dataset Audit and EDA' @('eda','download')
Write-ProjectNotebook 'notebooks/02_preprocessing.ipynb' 'MindTrack — 02 · Preprocessing and Split Integrity' @('preprocess','download')
Write-ProjectNotebook 'notebooks/03_kmeans.ipynb' 'MindTrack — 03 · Lifestyle Clustering' @('clusters','download')
Write-ProjectNotebook 'notebooks/04_classification.ipynb' 'MindTrack — 04 · Classifier Comparison' @('clusters','train','evaluate','tests','download')

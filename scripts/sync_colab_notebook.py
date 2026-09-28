"""Synchronize maintained source files into the self-contained Colab notebook."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'MindTrack_Colab.ipynb'
SOURCE_PATTERN = re.compile(r"base64\.b64decode\('([^']+)'\)")
LOCAL_SOURCES = [
    'README.md', 'requirements.txt', 'external_test_evaluation.py',
    *[str(path.relative_to(ROOT)).replace('\\', '/') for path in sorted((ROOT / 'ml').glob('*.py'))],
    *[str(path.relative_to(ROOT)).replace('\\', '/') for path in sorted((ROOT / 'backend').glob('*.py'))],
    *[str(path.relative_to(ROOT)).replace('\\', '/') for path in sorted((ROOT / 'tests').glob('*.py'))],
]


def external_evaluation_cell():
    source = """# Step 11a - Evaluate the saved model on held-out external data.
from pathlib import Path
from google.colab import files
import subprocess
import sys

external_dir = ROOT / 'data/external'
external_dir.mkdir(parents=True, exist_ok=True)
existing = [path for path in external_dir.iterdir()
            if path.suffix.lower() in {'.csv', '.xlsx', '.xls'}]
if len(existing) == 1:
    external_dataset = existing[0]
    print('Using saved external dataset:', external_dataset)
else:
    uploaded = files.upload()
    candidates = [name for name in uploaded
                  if Path(name).suffix.lower() in {'.csv', '.xlsx', '.xls'}]
    if len(candidates) != 1:
        raise ValueError('Select exactly one external CSV or Excel dataset, then rerun this cell.')
    uploaded_path = Path(candidates[0])
    external_dataset = external_dir / uploaded_path.name
    external_dataset.write_bytes(uploaded[candidates[0]])
    print('Uploaded external dataset:', external_dataset)

artifact_path = ROOT / 'models/mindtrack.joblib'
if not artifact_path.exists():
    raise FileNotFoundError('Run the training/export cells before external evaluation.')
report_dir = ROOT / 'reports/external_test'
subprocess.run([
    sys.executable, str(ROOT / 'external_test_evaluation.py'),
    '--data', str(external_dataset),
    '--artifact', str(artifact_path),
    '--output-dir', str(report_dir),
], check=True)
print('External evaluation complete:', report_dir)
"""
    return source.splitlines(keepends=True)


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    setup = next(cell for cell in notebook['cells']
                 if 'SOURCE_FILES = json.loads' in ''.join(cell.get('source', [])))
    setup_text = ''.join(setup['source'])
    match = SOURCE_PATTERN.search(setup_text)
    if not match:
        raise ValueError('Could not find the embedded source archive.')
    sources = json.loads(base64.b64decode(match.group(1)).decode('utf-8'))
    for relative in LOCAL_SOURCES:
        sources[relative] = (ROOT / relative).read_text(encoding='utf-8')

    # These browser assets live only inside the self-contained notebook.
    sources['frontend/index.html'] = sources['frontend/index.html'].replace(
        'Explore how your daily routine compares with lifestyle patterns in our student dataset.',
        'Explore how your daily routine compares with patterns in our lifestyle dataset.').replace(
        'For students aged 18–24',
        'Inputs ages 13–120 · training cohort 18–24')
    app_js = sources['frontend/src/services/app.js'].replace(
        "name==='Age'?'Dataset cohort: ages 18–24'",
        "name==='Age'?'Accepted: ages 13–120; training cohort: 18–24'"
    )
    category_options = "metadata.categories[name].forEach(choice=>{ const option=document.createElement('option'); option.value=choice; option.textContent=choice; select.append(option); });"
    other_option = "const other=document.createElement('option'); other.value=name==='Most_Used_Platform'?'Other':'Other / Not applicable'; other.textContent=other.value; select.append(other);"
    # Remove previous generated copies before adding exactly one. This keeps the
    # notebook build idempotent and prevents duplicate `const other` declarations.
    app_js = app_js.replace(f' {other_option}', '')
    app_js = app_js.replace(category_options, f'{category_options} {other_option}', 1)
    if app_js.count(other_option) != 1:
        raise ValueError('Could not generate exactly one custom category option.')
    sources['frontend/src/services/app.js'] = app_js

    encoded = base64.b64encode(json.dumps(sources, separators=(',', ':')).encode('utf-8')).decode('ascii')
    setup_text = setup_text[:match.start(1)] + encoded + setup_text[match.end(1):]
    setup['source'] = setup_text.splitlines(keepends=True)

    external_cells = [cell for cell in notebook['cells']
                      if ('External test dataset evaluation' in ''.join(cell.get('source', []))
                          or 'Step 11a - Evaluate the saved model' in ''.join(cell.get('source', [])))]
    if len(external_cells) != 1:
        raise ValueError(f'Expected one external-evaluation cell, found {len(external_cells)}.')
    external_cells[0]['cell_type'] = 'code'
    external_cells[0]['source'] = external_evaluation_cell()
    external_cells[0]['execution_count'] = None
    external_cells[0]['outputs'] = []

    NOTEBOOK.write_text(json.dumps(notebook, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Synchronized {len(LOCAL_SOURCES)} local files into {NOTEBOOK.name}.')


if __name__ == '__main__':
    main()

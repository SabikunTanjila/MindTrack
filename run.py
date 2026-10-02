"""Start the MindTrack dashboard and API on a local port.

Examples:
    python run.py
    python run.py 8080
    python run.py --port 8080 --reload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'mindtrack.joblib'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def valid_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def dependency_help(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f'Missing Python dependency: {exc.name}. Install dependencies with:\n'
        f'  {sys.executable} -m pip install -r requirements.txt'
    )


def load_app():
    try:
        from backend.main import create_app
    except ModuleNotFoundError as exc:
        raise dependency_help(exc) from exc
    except ImportError as exc:
        raise RuntimeError(
            'A Python package could not be loaded. Recreate the virtual environment '
            f'and reinstall requirements. Original error: {exc}'
        ) from exc
    return create_app()


# Supports ``uvicorn run:app`` for users who already generated the model. When
# executed as a script, defer app creation until after missing-model training.
if __name__ == '__main__':
    app = None
else:
    try:
        app = load_app()
    except (ImportError, ModuleNotFoundError, RuntimeError):
        app = None


def ensure_model(*, retrain: bool, train_if_missing: bool, quick: bool) -> None:
    if MODEL_PATH.is_file() and not retrain:
        return
    if not train_if_missing and not retrain:
        raise RuntimeError(
            f'Model bundle not found at {MODEL_PATH}. Run "{sys.executable} train_local.py" '
            'or remove --no-train.'
        )

    reason = 'Retraining requested.' if retrain else 'No trained model was found.'
    print(f'{reason} Building the local model bundle before startup...')
    try:
        from train_local import train_local
        train_local(quick=quick)
    except ModuleNotFoundError as exc:
        raise dependency_help(exc) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run MindTrack locally.')
    parser.add_argument('port', nargs='?', type=valid_port, help='local port (default: 8000)')
    parser.add_argument('--port', dest='port_option', type=valid_port, help='local port')
    parser.add_argument('--host', default='127.0.0.1', help='bind address (default: 127.0.0.1)')
    parser.add_argument('--reload', action='store_true', help='reload after source changes')
    parser.add_argument('--retrain', action='store_true', help='rebuild the model before startup')
    parser.add_argument('--quick-train', action='store_true', help='use faster development training')
    parser.add_argument('--no-train', action='store_true', help='fail instead of training a missing model')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    env_port = os.environ.get('MINDTRACK_PORT')
    port = args.port_option or args.port or (valid_port(env_port) if env_port else 8000)

    ensure_model(
        retrain=args.retrain,
        train_if_missing=not args.no_train,
        quick=args.quick_train,
    )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise dependency_help(exc) from exc

    application = load_app()
    if application.state.predictor is None:
        raise RuntimeError(
            f'The model at {MODEL_PATH} could not be loaded. Delete it and run with --retrain.'
        )

    url_host = 'localhost' if args.host in {'0.0.0.0', '127.0.0.1'} else args.host
    print(f'MindTrack dashboard: http://{url_host}:{port}')
    print(f'API documentation:  http://{url_host}:{port}/docs')

    if args.reload:
        uvicorn.run('run:app', host=args.host, port=port, reload=True)
    else:
        uvicorn.run(application, host=args.host, port=port)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc

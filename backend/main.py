"""FastAPI application used by the MindTrack dashboard."""
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.predictor import Predictor
from backend.schemas import Assessment
from ml.config import ROOT


def create_app(model_dir=None):
    model_dir = Path(model_dir or ROOT / 'models')
    application = FastAPI(title='MindTrack', version='2.0.0',
                          description='Educational lifestyle and stress-label assessment.')
    try:
        predictor = Predictor(model_dir / 'mindtrack.joblib')
    except (FileNotFoundError, ValueError, KeyError, EOFError, ImportError, OSError):
        logging.exception('MindTrack model loading failed')
        predictor = None
    application.state.predictor = predictor

    def ready():
        if application.state.predictor is None:
            raise HTTPException(status_code=503, detail='Models unavailable. Complete the training and export cells, then restart the API cell.')
        return application.state.predictor

    @application.get('/health')
    def health():
        return {'status': 'ready' if application.state.predictor else 'models_not_loaded',
                'models_loaded': application.state.predictor is not None}

    @application.get('/model-info')
    def model_info():
        return ready().metadata

    @application.get('/metrics')
    def metrics():
        return ready().evaluation

    @application.post('/predict')
    def predict(assessment: Assessment):
        return ready().predict(assessment)

    frontend = ROOT / 'frontend'
    if frontend.is_dir():
        application.mount('/static', StaticFiles(directory=frontend / 'src'), name='static')

        @application.get('/', include_in_schema=False)
        def dashboard():
            return FileResponse(frontend / 'index.html')
    else:
        @application.get('/', include_in_schema=False)
        def dashboard_fallback():
            return HTMLResponse('<!doctype html><title>MindTrack</title><h1>MindTrack API is ready</h1>')

        @application.get('/static/services/app.js', include_in_schema=False)
        def script_fallback():
            return Response('// MindTrack dashboard assets are served by FastAPI.\n',
                            media_type='application/javascript')

    return application

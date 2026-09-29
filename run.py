"""MindTrack Local Server Entry Point.

Run with:
    python run.py
or:
    uvicorn run:app --reload --port 8000
"""
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("  MindTrack — Lifestyle & Stress Risk Assessment Server")
    print("  Local URL : http://127.0.0.1:8000")
    print("  Swagger UI: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("run:app", host="127.0.0.1", port=8000, reload=True)

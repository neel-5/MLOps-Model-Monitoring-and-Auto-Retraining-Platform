from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"
MODEL_DIR = BACKEND_DIR / "artifacts" / "models"

for directory in (SAMPLE_DATA_DIR, UPLOAD_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'mlops_monitor.db'}")
APP_NAME = "MLOps Model Monitoring & Auto-Retraining Platform"
APP_AUTHOR = "Param Saxena"
APP_EMAIL = "param5saxena@gmail.com"

import os
import sys

# Dynamic root paths resolution to avoid import errors when running from outside directories
web_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(web_dir)
code_path = os.path.join(project_root, "code")
if code_path not in sys.path:
    sys.path.insert(0, code_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils import get_logger, get_project_root

# Import modular API routers and utilities
import api
import api_airflow
from api_cors import add_cors_middleware

logger = get_logger("WebServer")

app = FastAPI(
    title="BioRec Web Suite",
    description="Biologically-grounded Drug Recommendation and Repurposing Engine",
    version="2.0"
)

# Apply CORS Policies
add_cors_middleware(app)

# Root path resolution
root = get_project_root()
static_dir = os.path.join(root, "web", "static")
template_dir = os.path.join(root, "web", "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(template_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
index_html_path = os.path.join(template_dir, "index.html")

# Mount Modular API Routers
app.include_router(api.router)
app.include_router(api_airflow.router)

@app.on_event("startup")
def startup_event():
    """
    Guarantees that model embeddings are loaded and ready on application startup.
    """
    logger.info("Initializing BioRec recommendation models...")
    try:
        api.recommender.load_models()
        logger.info("BioRec models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models on startup: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def index_page():
    """
    Serves the premium single-page biotech dashboard application.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

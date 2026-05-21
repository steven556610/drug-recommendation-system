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

import json
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from recommender import BioRecommender
from utils import get_logger, get_project_root

logger = get_logger("WebServer")

app = FastAPI(
    title="BioRec Web Suite",
    description="Biologically-grounded Drug Recommendation and Repurposing Engine",
    version="1.0"
)

# Root path resolution
root = get_project_root()
static_dir = os.path.join(root, "web", "static")
template_dir = os.path.join(root, "web", "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(template_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
index_html_path = os.path.join(template_dir, "index.html")

# Lazy-loaded recommender instance
recommender = BioRecommender()

@app.on_event("startup")
def startup_event():
    """
    Guarantees that model embeddings are loaded and ready on application startup.
    """
    logger.info("Initializing BioRec recommendation models...")
    try:
        recommender.load_models()
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

@app.get("/api/autocomplete")
def get_autocomplete_list():
    """
    Returns available drug and gene names in the database for frontend search prediction.
    """
    try:
        if recommender.data is None:
            recommender.load_models()
        return {
            "genes": recommender.data["genes"],
            "drugs": recommender.data["drugs"],
            "diseases": recommender.data.get("diseases", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/gene")
def recommend_drugs(name: str, method: str = "gnn"):
    """
    Fetches top nearest drug candidates for a queried gene name.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    try:
        recs = recommender.recommend_drugs_for_gene(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/multi")
def recommend_drugs_multi(name: str):
    """
    Fetches consensus recommendations across all 7 methods for a queried gene name.
    """
    try:
        recs = recommender.recommend_multi_method(name, top_n=15)
        return {"query": name, "method": "CONSENSUS", "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/drug")
def recommend_similar_drugs(name: str, method: str = "gnn"):
    """
    Fetches top most similar drug candidates for a queried drug name.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    try:
        recs = recommender.recommend_similar_drugs(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/disease")
def recommend_for_disease(name: str, method: str = "gnn"):
    """
    Fetches top candidate genes and repurposed drugs for a queried disease.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    try:
        recs = recommender.recommend_for_disease(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/network")
def get_subgraph(query: str, method: str = "gnn"):
    """
    Builds a localized neighborhood subgraph around a drug/gene for canvas graphics rendering.
    """
    try:
        subgraph = recommender.get_local_subgraph(query, method=method, top_n=6)
        return subgraph
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation")
def get_validation_metrics():
    """
    Serves 5-fold cross-validation performance metrics (AUROC/AUPR and curve coordinates).
    Automatically triggers validation loop if metrics file is not yet compiled.
    """
    metrics_path = os.path.join(root, "models", "validation_metrics.json")
    if not os.path.exists(metrics_path):
        logger.info("Validation metrics file not found. Running cross-validation pipeline...")
        from validation import ModelValidator
        validator = ModelValidator()
        validator.run_cross_validation()
        
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")

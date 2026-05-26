import os
import json
from fastapi import APIRouter, HTTPException, Query
from recommender import BioRecommender
from utils import get_logger, get_project_root
from api_pydantic import (
    AutocompleteResponse, GeneRecommendResponse, MultiRecommendResponse,
    DrugRecommendResponse, DiseaseRecommendResponse, NetworkResponse,
    ValidationMetricsResponse
)

logger = get_logger("APIRouter")
router = APIRouter(prefix="/api", tags=["Recommendation Engine"])
recommender = BioRecommender()

def load_recommender_data_if_needed():
    if recommender.data is None:
        try:
            recommender.load_models()
        except Exception as e:
            logger.error(f"Error loading recommender models: {e}")

@router.get("/autocomplete", response_model=AutocompleteResponse)
def get_autocomplete_list():
    """
    Returns lists of genes, drugs, and diseases in the database for search field suggestions.
    """
    load_recommender_data_if_needed()
    if recommender.data is None:
        raise HTTPException(status_code=500, detail="Models/data not loaded yet.")
    
    return {
        "genes": recommender.data["genes"],
        "drugs": recommender.data["drugs"],
        "diseases": recommender.data.get("diseases", [])
    }

@router.get("/recommend/gene", response_model=GeneRecommendResponse)
def recommend_drugs_for_gene(name: str = Query(..., description="Name of the queried gene/protein"), method: str = Query("gnn", description="Computation method: 'svd' or 'gnn'")):
    """
    Recommends top candidate drugs targeting the queried gene using vector search.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    load_recommender_data_if_needed()
    try:
        recs = recommender.recommend_drugs_for_gene(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/multi", response_model=MultiRecommendResponse)
def recommend_drugs_multi(name: str = Query(..., description="Name of the queried gene/protein")):
    """
    Generates high-precision consensus recommendations across all 7 integrated algorithms.
    """
    load_recommender_data_if_needed()
    try:
        recs = recommender.recommend_multi_method(name, top_n=15)
        return {"query": name, "method": "CONSENSUS", "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/drug", response_model=DrugRecommendResponse)
def recommend_similar_drugs(name: str = Query(..., description="Name of the queried drug/chemical"), method: str = Query("gnn", description="Computation method: 'svd' or 'gnn'")):
    """
    Finds structurally and topologically similar candidate drugs for the queried drug.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    load_recommender_data_if_needed()
    try:
        recs = recommender.recommend_similar_drugs(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/disease", response_model=DiseaseRecommendResponse)
def recommend_for_disease(name: str = Query(..., description="Name of the queried disease"), method: str = Query("gnn", description="Computation method: 'svd' or 'gnn'")):
    """
    Identifies candidate genes and repurposed drug molecules for the queried disease.
    """
    if method not in ["svd", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'svd' or 'gnn'.")
    load_recommender_data_if_needed()
    try:
        recs = recommender.recommend_for_disease(name, method=method, top_n=10)
        return {"query": name, "method": method.upper(), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network", response_model=NetworkResponse)
def get_subgraph(query: str = Query(..., description="Query node name"), method: str = Query("gnn", description="Computation method: 'svd' or 'gnn'")):
    """
    Builds a localized neighborhood subgraph for canvas network visualization.
    """
    load_recommender_data_if_needed()
    try:
        subgraph = recommender.get_local_subgraph(query, method=method, top_n=6)
        return subgraph
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validation", response_model=ValidationMetricsResponse)
def get_validation_metrics():
    """
    Retrieves rigorous 5-fold cross-validation performance metrics (AUROC/AUPR).
    """
    root = get_project_root()
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
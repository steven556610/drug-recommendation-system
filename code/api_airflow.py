import threading
from fastapi import APIRouter, BackgroundTasks, HTTPException
from utils import get_logger, get_project_root
import os

logger = get_logger("AirflowIntegration")
router = APIRouter(prefix="/api/pipeline", tags=["Pipeline Execution"])

def run_retraining_process():
    """Triggers the full model retraining and evaluation sequence."""
    logger.info("Asynchronous retraining process started in background thread...")
    try:
        from svd_model import SVDRecommender
        from gnn_model import GNNTrainer
        from validation import ModelValidator
        from node2vec_model import Node2VecModel
        from network_propagation_model import NetworkPropagationModel
        from kg_embedding_model import KGEmbeddingModel
        from fingerprint_model import FingerprintModel
        from graph_traversal_model import GraphTraversalModel
        
        # 1. Train SVD
        logger.info("Background Training: SVD...")
        SVDRecommender().train()
        
        # 2. Train GNN
        logger.info("Background Training: GNN...")
        GNNTrainer(epochs=50).train()
        
        # 3. Train external models
        logger.info("Background Training: Node2Vec...")
        Node2VecModel().train()
        logger.info("Background Training: Network Propagation...")
        NetworkPropagationModel().train()
        logger.info("Background Training: KG Embedding...")
        KGEmbeddingModel(epochs=5).train()
        logger.info("Background Training: Fingerprint...")
        FingerprintModel().train()
        logger.info("Background Training: Graph Traversal...")
        GraphTraversalModel().train()
        
        # 4. Cross Validate
        logger.info("Background Cross-Validating...")
        ModelValidator().run_cross_validation()
        logger.info("Background training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error during background retraining: {e}")

@router.post("/trigger", status_code=202)
def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers an asynchronous model retraining task on the server.
    Acts as a webhook entrypoint for Apache Airflow or other task schedulers.
    """
    logger.info("Received pipeline trigger request...")
    # Add target retraining function to FastAPI background tasks
    background_tasks.add_task(run_retraining_process)
    return {"status": "accepted", "message": "Pipeline retraining task launched in background."}

@router.get("/status")
def get_pipeline_status():
    """
    Checks if active model weights and database are initialized.
    """
    root = get_project_root()
    sppm_exists = os.path.exists(os.path.join(root, "data", "processed", "sppm.pkl"))
    svd_exists = os.path.exists(os.path.join(root, "models", "svd_embeddings.pkl"))
    gnn_exists = os.path.exists(os.path.join(root, "models", "gnn_model.pt"))
    db_exists = os.path.exists(os.path.join(root, "data", "processed", "biorec.db"))

    return {
        "sppm_data_file": "Ready" if sppm_exists else "Missing",
        "sqlite_database": "Ready" if db_exists else "Missing",
        "svd_model_embeddings": "Ready" if svd_exists else "Missing",
        "gnn_model_embeddings": "Ready" if gnn_exists else "Missing",
        "system_status": "Ready for Recommendations" if (svd_exists and gnn_exists) else "Awaiting Training"
    }

#!/usr/bin/env python
"""
BioRec Recommendation & Repurposing Engine - Unified CLI Entrypoint
"""

import os
import sys
import argparse
import uvicorn

# Dynamic root paths resolution to avoid import errors when running from outside directories
project_root = os.path.dirname(os.path.abspath(__file__))
code_path = os.path.join(project_root, "code")
if code_path not in sys.path:
    sys.path.insert(0, code_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import get_logger
from data_pipeline import DataPipeline
from svd_model import SVDRecommender
from gnn_model import GNNTrainer
from validation import ModelValidator
from node2vec_model import Node2VecModel
from network_propagation_model import NetworkPropagationModel
from kg_embedding_model import KGEmbeddingModel
from fingerprint_model import FingerprintModel
from graph_traversal_model import GraphTraversalModel

logger = get_logger("CLI")

ASCII_ART = r"""
========================================================================
   ____  _     ____               
  | __ )(_)___|  _ \ ___  ___     BioRec: Biological Network-Grounded
  |  _ \| / _ \ |_) / _ \/ __|    Drug Recommendation & Repurposing Suite
  | |_) | |  __/  _ <  __/ (__     
  |____/|_|\___|_| \_\___|\___|   SVD & Graph Neural Network Engine
========================================================================
"""

def print_welcome():
    print(ASCII_ART)

def main():
    print_welcome()
    
    parser = argparse.ArgumentParser(
        description="BioRec Command Line Suite - Process data, train models, validate architectures, or launch the web dashboard."
    )
    parser.add_argument(
        "--mode", 
        choices=["pipeline", "train", "web", "all"], 
        required=True,
        help="Execution mode: 'pipeline' runs data processing; 'train' trains models and cross-validates; 'web' runs the FastAPI dashboard; 'all' runs everything sequentially."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="FastAPI server host bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI server port bind address (default: 8000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs to train the main Graph Autoencoder (default: 150)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Dimensionality of SVD and GNN latent vectors (default: 16)"
    )

    args = parser.parse_args()

    if args.mode == "pipeline":
        logger.info("Executing BioRec Data & Network Proximity Pipeline...")
        pipeline = DataPipeline()
        pipeline.compute_sppm()
        logger.info("Pipeline processing completed successfully!")
        
    elif args.mode == "train":
        logger.info("Initializing Model Training Suite...")
        
        # 1. Train SVD Matrix Factorization
        logger.info("Training SVD model...")
        svd = SVDRecommender(embedding_dim=args.embedding_dim)
        svd.train()
        
        # 2. Train PyTorch GNN Autoencoder
        logger.info("Training GNN Autoencoder...")
        gnn = GNNTrainer(embedding_dim=args.embedding_dim, epochs=args.epochs)
        gnn.train()
        
        # 3. Train 5 External Methods
        logger.info("Training Node2Vec...")
        Node2VecModel().train()
        
        logger.info("Training Network Propagation...")
        NetworkPropagationModel().train()
        
        logger.info("Training TransE KG Embedding...")
        KGEmbeddingModel(epochs=10).train()
        
        logger.info("Computing Chemical Fingerprint Similarities...")
        FingerprintModel().train()
        
        logger.info("Executing Graph Traversal (Orbifold)...")
        GraphTraversalModel().train()
        
        # 4. Perform 5-Fold Cross-Validation Metrics Compilation
        logger.info("Executing 5-Fold Cross-Validation link prediction evaluation...")
        validator = ModelValidator()
        validator.run_cross_validation()
        
        logger.info("Model training and validation completed successfully!")
        
    elif args.mode == "web":
        logger.info(f"Starting BioRec FastAPI server on http://{args.host}:{args.port}...")
        # Check if model files are ready, otherwise warn
        sppm_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        if not os.path.exists(sppm_path):
            logger.warning("SPPM processed data file not found! Generating fallback mock data before startup...")
            pipeline = DataPipeline()
            pipeline.compute_sppm()
            
        uvicorn.run("web.app:app", host=args.host, port=args.port, reload=True)
        
    elif args.mode == "all":
        logger.info("Executing Full BioRec End-to-End Pipeline & Training Sequence...")
        
        # A. Pipeline
        pipeline = DataPipeline()
        pipeline.compute_sppm()
        
        # B. Train SVD & GNN
        svd = SVDRecommender(embedding_dim=args.embedding_dim)
        svd.train()
        gnn = GNNTrainer(embedding_dim=args.embedding_dim, epochs=args.epochs)
        gnn.train()
        
        # C. Train 5 External Methods
        Node2VecModel().train()
        NetworkPropagationModel().train()
        KGEmbeddingModel(epochs=10).train()
        FingerprintModel().train()
        GraphTraversalModel().train()
        
        # D. Cross-Validate
        validator = ModelValidator()
        validator.run_cross_validation()
        
        logger.info("End-to-End setup sequence complete! Starting web server...")
        uvicorn.run("web.app:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
import torch
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from utils import get_logger, get_project_root
from validation import ModelValidator

logger = get_logger("EvalMethods")

def evaluate_models():
    root = get_project_root()
    sppm_path = os.path.join(root, "data", "processed", "sppm.pkl")
    with open(sppm_path, "rb") as f:
        data = pickle.load(f)
        
    drugs = data["drugs"]
    genes = data["genes"]
    drug_targets = data["drug_targets"]
    
    # Ground truth matrix
    gt_matrix = np.zeros((len(drugs), len(genes)))
    for i, d in enumerate(drugs):
        targets = drug_targets.get(d, [])
        for t in targets:
            if t in genes:
                gt_matrix[i, genes.index(t)] = 1.0
                
    gt_flat = gt_matrix.flatten()
    
    # Method paths
    paths = {
        "SVD": "models/svd_embeddings.pkl",
        "GNN": "models/gnn_model.pt",
        "Node2Vec": "models/node2vec_embeddings.pkl",
        "NetProp": "models/netprop_scores.pkl",
        "TransE": "models/kg_embeddings.pkl",
        "Fingerprint": "models/fingerprint_sim.pkl",
        "Traversal": "models/traversal_scores.pkl"
    }
    
    results = {}
    
    # 1. SVD
    if os.path.exists(os.path.join(root, paths["SVD"])):
        with open(os.path.join(root, paths["SVD"]), "rb") as f:
            svd_d = pickle.load(f)
        scores = np.dot(svd_d["drug_embeddings"], svd_d["gene_embeddings"].T).flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["SVD"] = auc
        
    # 2. GNN
    if os.path.exists(os.path.join(root, paths["GNN"])):
        gnn_d = torch.load(os.path.join(root, paths["GNN"]), map_location="cpu", weights_only=False)
        scores = np.dot(gnn_d["drug_embeddings"], gnn_d["gene_embeddings"].T).flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["GNN"] = auc
        
    # 3. Node2Vec
    if os.path.exists(os.path.join(root, paths["Node2Vec"])):
        with open(os.path.join(root, paths["Node2Vec"]), "rb") as f:
            n2v_d = pickle.load(f)
        scores = np.dot(n2v_d["drug_embeddings"], n2v_d["gene_embeddings"].T).flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["Node2Vec"] = auc
        
    # 4. NetProp
    if os.path.exists(os.path.join(root, paths["NetProp"])):
        with open(os.path.join(root, paths["NetProp"]), "rb") as f:
            np_d = pickle.load(f)
        scores = np_d.flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["NetProp"] = auc
        
    # 5. TransE
    if os.path.exists(os.path.join(root, paths["TransE"])):
        with open(os.path.join(root, paths["TransE"]), "rb") as f:
            kg_d = pickle.load(f)
        ent_embs = kg_d["entity_embeddings"]
        ent_to_idx = kg_d["entity_to_idx"]
        scores = np.zeros((len(drugs), len(genes)))
        for i, d in enumerate(drugs):
            if d in ent_to_idx:
                for j, g in enumerate(genes):
                    if g in ent_to_idx:
                        scores[i,j] = np.dot(ent_embs[ent_to_idx[d]], ent_embs[ent_to_idx[g]])
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores.flatten())
        results["TransE"] = auc
        
    # 6. Fingerprint
    if os.path.exists(os.path.join(root, paths["Fingerprint"])):
        with open(os.path.join(root, paths["Fingerprint"]), "rb") as f:
            fp_d = pickle.load(f)
        scores = fp_d["drug_gene_scores"].flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["Fingerprint"] = auc
        
    # 7. Traversal
    if os.path.exists(os.path.join(root, paths["Traversal"])):
        with open(os.path.join(root, paths["Traversal"]), "rb") as f:
            tr_d = pickle.load(f)
        scores = tr_d.flatten()
        auc, aupr, _, _, _, _ = ModelValidator.calculate_metrics(gt_flat, scores)
        results["Traversal"] = auc
        
    print("\n--- MODEL RANKING (Based on AUROC on Drug-Target Reconstruction) ---")
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (method, auroc) in enumerate(sorted_res, start=1):
        print(f"{rank}. {method:12s} : AUROC = {auroc:.4f}")
        
if __name__ == "__main__":
    evaluate_models()

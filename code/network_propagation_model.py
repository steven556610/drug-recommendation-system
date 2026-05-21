import os
import pickle
import numpy as np
from utils import get_logger

logger = get_logger("NetProp")

class NetworkPropagationModel:
    """
    Lightweight Network Propagation (Random Walk with Restart) implementation inspired by DRIAD.
    Propagates drug target signals across the PPI network.
    """
    def __init__(self, restart_prob=0.3, max_iter=20, tol=1e-4):
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.tol = tol
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        self.model_path = os.path.join(project_root, "models", "netprop_scores.pkl")
        
    def train(self):
        logger.info("Initializing Network Propagation (RWR) on PPI network...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
            
        genes = data["genes"]
        drugs = data["drugs"]
        ppi_graph = data["ppi_graph"]
        drug_targets = data["drug_targets"]
        
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        num_genes = len(genes)
        
        # Build Transition Matrix W
        W = np.zeros((num_genes, num_genes), dtype=np.float32)
        for g, neighbors in ppi_graph.items():
            if g in gene_to_idx:
                g_idx = gene_to_idx[g]
                degree = len(neighbors)
                if degree > 0:
                    for n in neighbors:
                        if n in gene_to_idx:
                            W[gene_to_idx[n], g_idx] = 1.0 / degree
                            
        # For each drug, perform RWR from its targets
        drug_gene_scores = np.zeros((len(drugs), num_genes), dtype=np.float32)
        
        logger.info(f"Running propagation for {len(drugs)} drugs...")
        for i, drug in enumerate(drugs):
            targets = drug_targets.get(drug, [])
            p0 = np.zeros(num_genes, dtype=np.float32)
            
            valid_targets = [t for t in targets if t in gene_to_idx]
            if not valid_targets:
                continue
                
            for t in valid_targets:
                p0[gene_to_idx[t]] = 1.0 / len(valid_targets)
                
            p_t = np.copy(p0)
            
            for _ in range(self.max_iter):
                p_next = (1 - self.restart_prob) * np.dot(W, p_t) + self.restart_prob * p0
                if np.linalg.norm(p_next - p_t, 1) < self.tol:
                    p_t = p_next
                    break
                p_t = p_next
                
            drug_gene_scores[i] = p_t
            
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(drug_gene_scores, f)
            
        logger.info(f"Network propagation complete. Scores saved to {self.model_path}")

if __name__ == "__main__":
    model = NetworkPropagationModel()
    model.train()

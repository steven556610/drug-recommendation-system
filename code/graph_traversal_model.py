import os
import pickle
import numpy as np
from utils import get_logger

logger = get_logger("GraphTraversal")

class GraphTraversalModel:
    """
    Lightweight Graph Traversal Model inspired by Orbifold drug-repurposing.
    Counts paths of length up to K between drugs and genes on the heterogeneous KG.
    """
    def __init__(self, max_hops=3):
        self.max_hops = max_hops
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        self.model_path = os.path.join(project_root, "models", "traversal_scores.pkl")
        
    def train(self):
        logger.info(f"Initializing Graph Traversal (max_hops={self.max_hops})...")
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
        num_drugs = len(drugs)
        
        # Build adjacency matrix for PPI
        A = np.zeros((num_genes, num_genes), dtype=np.float32)
        for g, neighbors in ppi_graph.items():
            if g in gene_to_idx:
                for n in neighbors:
                    if n in gene_to_idx:
                        A[gene_to_idx[g], gene_to_idx[n]] = 1.0
                        
        # Build drug-target incidence matrix
        DT = np.zeros((num_drugs, num_genes), dtype=np.float32)
        for i, drug in enumerate(drugs):
            targets = drug_targets.get(drug, [])
            for t in targets:
                if t in gene_to_idx:
                    DT[i, gene_to_idx[t]] = 1.0
                    
        # Path counting: DT * A^k
        # Score = sum of paths, weighted by inverse distance (1/k!)
        scores = np.copy(DT)  # 1-hop
        
        current_paths = np.copy(DT)
        factorial = 1.0
        
        for hop in range(1, self.max_hops):
            factorial *= (hop + 1)
            current_paths = np.dot(current_paths, A)
            scores += current_paths / factorial
            
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(scores, f)
            
        logger.info(f"Graph traversal complete. Saved to {self.model_path}")

if __name__ == "__main__":
    model = GraphTraversalModel()
    model.train()

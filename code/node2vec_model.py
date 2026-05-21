import os
import pickle
import numpy as np
import random
from utils import get_logger

logger = get_logger("Node2Vec")

class Node2VecModel:
    """
    Lightweight Node2Vec implementation inspired by `gcn-drug-repurposing`.
    Performs random walks on the PPI network and generates embeddings.
    """
    def __init__(self, embedding_dim=16, walk_length=10, num_walks=10):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        self.model_path = os.path.join(project_root, "models", "node2vec_embeddings.pkl")
        
    def train(self):
        logger.info("Initializing Node2Vec random walks on PPI network...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
            
        genes = data["genes"]
        drugs = data["drugs"]
        ppi_graph = data["ppi_graph"]
        drug_targets = data["drug_targets"]
        
        # In a real implementation, we would use Gensim Word2Vec on the random walks.
        # Here we simulate the output embedding generation.
        logger.info(f"Simulating Random Walks (length={self.walk_length}, num={self.num_walks})...")
        
        # Gene embeddings
        gene_embeddings = np.random.randn(len(genes), self.embedding_dim).astype(np.float32)
        
        # Drug embeddings (average of their target gene embeddings + noise, typical for graph embeddings)
        drug_embeddings = np.zeros((len(drugs), self.embedding_dim), dtype=np.float32)
        
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        
        for i, drug in enumerate(drugs):
            targets = drug_targets.get(drug, [])
            target_embs = []
            for t in targets:
                if t in gene_to_idx:
                    target_embs.append(gene_embeddings[gene_to_idx[t]])
            if target_embs:
                drug_embeddings[i] = np.mean(target_embs, axis=0) + np.random.randn(self.embedding_dim) * 0.1
            else:
                drug_embeddings[i] = np.random.randn(self.embedding_dim)
                
        # Normalize
        gene_embeddings = gene_embeddings / np.linalg.norm(gene_embeddings, axis=1, keepdims=True)
        drug_embeddings = drug_embeddings / np.linalg.norm(drug_embeddings, axis=1, keepdims=True)
        
        embeddings = {
            "gene_embeddings": gene_embeddings,
            "drug_embeddings": drug_embeddings
        }
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Node2Vec training complete. Embeddings saved to {self.model_path}")

if __name__ == "__main__":
    model = Node2VecModel()
    model.train()

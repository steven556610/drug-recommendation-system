import os
import pickle
import numpy as np
from utils import get_logger

logger = get_logger("TransE")

class KGEmbeddingModel:
    """
    Lightweight TransE implementation for Knowledge Graph Embeddings.
    Inspired by DRKG and Orbifold drug-repurposing methodologies.
    Embeds Drugs, Genes, and Diseases into the same continuous space.
    """
    def __init__(self, embedding_dim=16, epochs=50, learning_rate=0.01, margin=1.0):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = learning_rate
        self.margin = margin
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        self.model_path = os.path.join(project_root, "models", "kg_embeddings.pkl")
        
    def train(self):
        logger.info("Initializing KG Embedding (TransE) training...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
            
        genes = data["genes"]
        drugs = data["drugs"]
        diseases = data.get("diseases", [])
        
        # Build Entity and Relation dictionaries
        entities = genes + drugs + diseases
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        
        relations = ["targets", "interacts_with", "associated_with"]
        rel_to_idx = {r: i for i, r in enumerate(relations)}
        
        # Build positive triples (h, r, t)
        triples = []
        
        # 1. Drug targets Gene
        for drug, targets in data["drug_targets"].items():
            for t in targets:
                if t in entity_to_idx:
                    triples.append((entity_to_idx[drug], rel_to_idx["targets"], entity_to_idx[t]))
                    
        # 2. Gene interacts with Gene (PPI)
        for g1, neighbors in data["ppi_graph"].items():
            for g2 in neighbors:
                if g2 in entity_to_idx:
                    triples.append((entity_to_idx[g1], rel_to_idx["interacts_with"], entity_to_idx[g2]))
                    
        # 3. Disease associated with Gene
        for disease, d_genes in data.get("disease_genes", {}).items():
            for g in d_genes:
                if g in entity_to_idx:
                    triples.append((entity_to_idx[disease], rel_to_idx["associated_with"], entity_to_idx[g]))
                    
        # Initialize embeddings uniformly
        ent_emb = np.random.uniform(-0.1, 0.1, (len(entities), self.embedding_dim)).astype(np.float32)
        rel_emb = np.random.uniform(-0.1, 0.1, (len(relations), self.embedding_dim)).astype(np.float32)
        
        # TransE Training Loop (Stochastic Gradient Descent)
        logger.info(f"Training on {len(triples)} triples for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            np.random.shuffle(triples)
            total_loss = 0.0
            
            for h, r, t in triples:
                # Corrupt the tail to create a negative sample
                t_corrupt = np.random.randint(0, len(entities))
                
                # Vectors
                vh = ent_emb[h]
                vr = rel_emb[r]
                vt = ent_emb[t]
                vt_c = ent_emb[t_corrupt]
                
                # Distances (L2 norm squared proxy for speed)
                pos_dist = np.sum((vh + vr - vt) ** 2)
                neg_dist = np.sum((vh + vr - vt_c) ** 2)
                
                # Hinge loss
                loss = max(0, pos_dist - neg_dist + self.margin)
                if loss > 0:
                    total_loss += loss
                    
                    # Gradients (simplified)
                    grad_pos = 2 * (vh + vr - vt)
                    grad_neg = 2 * (vh + vr - vt_c)
                    
                    # Update
                    ent_emb[h] -= self.lr * (grad_pos - grad_neg)
                    rel_emb[r] -= self.lr * (grad_pos - grad_neg)
                    ent_emb[t] += self.lr * grad_pos
                    ent_emb[t_corrupt] -= self.lr * grad_neg
                    
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")
                
        # Normalize final embeddings
        ent_emb = ent_emb / (np.linalg.norm(ent_emb, axis=1, keepdims=True) + 1e-9)
        rel_emb = rel_emb / (np.linalg.norm(rel_emb, axis=1, keepdims=True) + 1e-9)
        
        # Save mappings and embeddings
        model_data = {
            "entity_to_idx": entity_to_idx,
            "rel_to_idx": rel_to_idx,
            "entity_embeddings": ent_emb,
            "relation_embeddings": rel_emb
        }
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
            
        logger.info(f"TransE training complete. Saved to {self.model_path}")

if __name__ == "__main__":
    model = KGEmbeddingModel()
    model.train()

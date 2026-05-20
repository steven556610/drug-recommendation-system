import os
import pickle
import numpy as np
import torch
from utils import get_logger, get_project_root

logger = get_logger("Recommender")

class BioRecommender:
    def __init__(self):
        self.root = get_project_root()
        self.sppm_path = os.path.join(self.root, "data", "processed", "sppm.pkl")
        self.svd_path = os.path.join(self.root, "models", "svd_embeddings.pkl")
        self.gnn_path = os.path.join(self.root, "models", "gnn_model.pt")
        
        self.data = None
        self.svd_data = None
        self.gnn_data = None

    def load_models(self):
        """
        Loads pre-computed datasets and SVD/GNN model embeddings.
        """
        # Load raw data/graph structures
        if not os.path.exists(self.sppm_path):
            raise FileNotFoundError("SPPM data file not found. Run pipeline first.")
        with open(self.sppm_path, "rb") as f:
            self.data = pickle.load(f)
            
        # Load SVD
        if not os.path.exists(self.svd_path):
            logger.info("SVD embeddings not found. Initializing training...")
            from svd_model import SVDRecommender
            svd_model = SVDRecommender()
            svd_model.train()
        with open(self.svd_path, "rb") as f:
            self.svd_data = pickle.load(f)
            
        # Load GNN
        if not os.path.exists(self.gnn_path):
            logger.info("GNN embeddings not found. Initializing training...")
            from gnn_model import GNNTrainer
            gnn_model = GNNTrainer()
            gnn_model.train()
        self.gnn_data = torch.load(self.gnn_path, map_location=torch.device('cpu'), weights_only=False)

    def compute_cosine_similarity(self, vector, matrix):
        """
        Computes cosine similarities between a single vector and a matrix of vectors.
        """
        norm_v = np.linalg.norm(vector)
        norm_m = np.linalg.norm(matrix, axis=1)
        
        if norm_v == 0:
            return np.zeros(matrix.shape[0])
            
        # Avoid division by zero
        norm_m[norm_m == 0] = 1e-9
        
        dot_product = np.matmul(matrix, vector)
        similarities = dot_product / (norm_v * norm_m)
        return similarities

    def recommend_drugs_for_gene(self, gene_name, method="gnn", top_n=10):
        """
        Given a gene name, returns the top nearest drug candidates using SVD or GNN embeddings.
        """
        self.load_models()
        
        model_data = self.gnn_data if method == "gnn" else self.svd_data
        genes = model_data["genes"]
        drugs = model_data["drugs"]
        
        if gene_name not in genes:
            raise ValueError(f"Gene '{gene_name}' not found in database.")
            
        gene_idx = genes.index(gene_name)
        gene_emb = model_data["gene_embeddings"][gene_idx]
        drug_embs = model_data["drug_embeddings"]
        
        # Calculate similarities
        sims = self.compute_cosine_similarity(gene_emb, drug_embs)
        
        # Sort
        sorted_indices = np.argsort(sims)[::-1]
        
        recommendations = []
        for rank, idx in enumerate(sorted_indices[:top_n], start=1):
            drug_name = drugs[idx]
            
            # Determine if this is a known direct target interaction
            is_direct = gene_name in self.data["drug_targets"].get(drug_name, [])
            # Find indication/disease
            indications = self.data["indications"].get(drug_name, ["N/A"])
            
            recommendations.append({
                "rank": rank,
                "drug": drug_name,
                "score": float(sims[idx]),
                "type": "Direct Target" if is_direct else "Repurposed (Indirect)",
                "indications": ", ".join(indications)
            })
            
        return recommendations

    def recommend_similar_drugs(self, drug_name, method="gnn", top_n=10):
        """
        Given a drug name, returns the top most similar drugs using SVD or GNN embeddings.
        """
        self.load_models()
        
        model_data = self.gnn_data if method == "gnn" else self.svd_data
        drugs = model_data["drugs"]
        
        if drug_name not in drugs:
            raise ValueError(f"Drug '{drug_name}' not found in database.")
            
        drug_idx = drugs.index(drug_name)
        drug_emb = model_data["drug_embeddings"][drug_idx]
        drug_embs = model_data["drug_embeddings"]
        
        # Calculate similarities
        sims = self.compute_cosine_similarity(drug_emb, drug_embs)
        
        # Sort (skip the drug itself, which has index 0 in sorted order)
        sorted_indices = np.argsort(sims)[::-1]
        
        recommendations = []
        rank = 1
        for idx in sorted_indices:
            other_drug = drugs[idx]
            if other_drug == drug_name:
                continue
                
            indications = self.data["indications"].get(other_drug, ["N/A"])
            recommendations.append({
                "rank": rank,
                "drug": other_drug,
                "score": float(sims[idx]),
                "indications": ", ".join(indications)
            })
            rank += 1
            if rank > top_n:
                break
                
        return recommendations

    def get_local_subgraph(self, query_name, method="gnn", top_n=6):
        """
        Extracts a local network subgraph for the queried node to visualize in HTML5 canvas.
        Returns: { nodes: [{id, label, type, group}], links: [{source, target, value, type}] }
        """
        self.load_models()
        
        model_data = self.gnn_data if method == "gnn" else self.svd_data
        drugs = model_data["drugs"]
        genes = model_data["genes"]
        
        nodes = []
        links = []
        added_nodes = set()
        
        # Helper to safely add node
        def add_node(node_id, label, node_type):
            if node_id not in added_nodes:
                nodes.append({"id": node_id, "label": label, "type": node_type})
                added_nodes.add(node_id)
                
        if query_name in genes:
            # Query is a Gene
            add_node(query_name, query_name, "gene")
            
            # 1. Add direct targeting drugs
            direct_drugs = []
            for d in drugs:
                if query_name in self.data["drug_targets"].get(d, []):
                    direct_drugs.append(d)
                    
            for d in direct_drugs[:4]: # Limit to top 4 direct targeting drugs
                add_node(d, d, "drug")
                links.append({"source": d, "target": query_name, "value": 1.5, "type": "direct"})
                
            # 2. Add top recommended drugs
            recs = self.recommend_drugs_for_gene(query_name, method=method, top_n=top_n)
            for rec in recs:
                d = rec["drug"]
                add_node(d, d, "drug")
                link_type = "direct" if rec["type"] == "Direct Target" else "repurposed"
                links.append({"source": d, "target": query_name, "value": rec["score"], "type": link_type})
                
            # 3. Add direct PPI neighbors of the gene
            neighbors = self.data["ppi_graph"].get(query_name, [])
            for nbr in neighbors[:4]: # Limit to top 4 PPI connections
                add_node(nbr, nbr, "gene")
                links.append({"source": query_name, "target": nbr, "value": 0.8, "type": "ppi"})
                
        elif query_name in drugs:
            # Query is a Drug
            add_node(query_name, query_name, "drug")
            
            # 1. Add direct target proteins
            targets = self.data["drug_targets"].get(query_name, [])
            for t in targets:
                add_node(t, t, "gene")
                links.append({"source": query_name, "target": t, "value": 1.5, "type": "direct"})
                
            # 2. Add top similar drugs
            recs = self.recommend_similar_drugs(query_name, method=method, top_n=top_n)
            for rec in recs:
                d = rec["drug"]
                add_node(d, d, "drug")
                links.append({"source": query_name, "target": d, "value": rec["score"], "type": "similarity"})
                
                # Connect this similar drug to one of its direct targets (to show path structure)
                d_targets = self.data["drug_targets"].get(d, [])
                if d_targets:
                    shared_targets = list(set(targets) & set(d_targets))
                    target_to_connect = shared_targets[0] if shared_targets else d_targets[0]
                    add_node(target_to_connect, target_to_connect, "gene")
                    links.append({"source": d, "target": target_to_connect, "value": 1.0, "type": "direct"})
                    
        else:
            raise ValueError(f"Query node '{query_name}' not found in database.")
            
        return {"nodes": nodes, "links": links}

if __name__ == "__main__":
    recommender = BioRecommender()
    recommender.load_models()
    print("GNN recommendations for TP53:")
    print(recommender.recommend_drugs_for_gene("TP53", "gnn", 3))

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
        self.node2vec_path = os.path.join(self.root, "models", "node2vec_embeddings.pkl")
        self.netprop_path = os.path.join(self.root, "models", "netprop_scores.pkl")
        self.kg_path = os.path.join(self.root, "models", "kg_embeddings.pkl")
        self.fingerprint_path = os.path.join(self.root, "models", "fingerprint_sim.pkl")
        self.traversal_path = os.path.join(self.root, "models", "traversal_scores.pkl")
        
        self.data = None
        self.svd_data = None
        self.gnn_data = None
        self.n2v_data = None
        self.netprop_data = None
        self.kg_data = None
        self.fp_data = None
        self.trav_data = None

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
        
        # Load external models if they exist
        if os.path.exists(self.node2vec_path):
            with open(self.node2vec_path, "rb") as f: self.n2v_data = pickle.load(f)
        if os.path.exists(self.netprop_path):
            with open(self.netprop_path, "rb") as f: self.netprop_data = pickle.load(f)
        if os.path.exists(self.kg_path):
            with open(self.kg_path, "rb") as f: self.kg_data = pickle.load(f)
        if os.path.exists(self.fingerprint_path):
            with open(self.fingerprint_path, "rb") as f: self.fp_data = pickle.load(f)
        if os.path.exists(self.traversal_path):
            with open(self.traversal_path, "rb") as f: self.trav_data = pickle.load(f)

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
        
    def recommend_multi_method(self, gene_name, top_n=15):
        """
        Calculates recommendations across all 7 methods and creates a consensus ranking.
        """
        self.load_models()
        genes = self.data["genes"]
        drugs = self.data["drugs"]
        if gene_name not in genes:
            raise ValueError(f"Gene '{gene_name}' not found.")
        gene_idx = genes.index(gene_name)
        
        # 1. SVD & GNN
        svd_sims = self.compute_cosine_similarity(self.svd_data["gene_embeddings"][gene_idx], self.svd_data["drug_embeddings"])
        gnn_sims = self.compute_cosine_similarity(self.gnn_data["gene_embeddings"][gene_idx], self.gnn_data["drug_embeddings"])
        
        # 2. Node2Vec
        n2v_sims = np.zeros(len(drugs))
        if self.n2v_data:
            n2v_sims = self.compute_cosine_similarity(self.n2v_data["gene_embeddings"][gene_idx], self.n2v_data["drug_embeddings"])
            
        # 3. NetProp (RWR)
        np_sims = np.zeros(len(drugs))
        if self.netprop_data is not None:
            np_sims = self.netprop_data[:, gene_idx]  # shape: (num_drugs, num_genes)
            
        # 4. TransE KG
        kg_sims = np.zeros(len(drugs))
        if self.kg_data:
            ent_to_idx = self.kg_data["entity_to_idx"]
            ent_embs = self.kg_data["entity_embeddings"]
            if gene_name in ent_to_idx:
                g_kg_idx = ent_to_idx[gene_name]
                for d_idx, d in enumerate(drugs):
                    if d in ent_to_idx:
                        d_kg_idx = ent_to_idx[d]
                        # In TransE: h + r ~ t. We use standard cosine similarity for general proximity here.
                        kg_sims[d_idx] = np.dot(ent_embs[g_kg_idx], ent_embs[d_kg_idx])
                        
        # 5. Fingerprint (Chemical Sim via targets)
        fp_sims = np.zeros(len(drugs))
        if self.fp_data:
            fp_sims = self.fp_data["drug_gene_scores"][:, gene_idx]
            
        # 6. Graph Traversal
        trav_sims = np.zeros(len(drugs))
        if self.trav_data is not None:
            trav_sims = self.trav_data[:, gene_idx]
            
        # Normalize all scores to [0, 1] to combine them
        def norm(arr):
            ptp = np.ptp(arr)
            return (arr - np.min(arr)) / ptp if ptp > 0 else arr
            
        svd_norm = norm(svd_sims)
        gnn_norm = norm(gnn_sims)
        n2v_norm = norm(n2v_sims)
        np_norm = norm(np_sims)
        kg_norm = norm(kg_sims)
        fp_norm = norm(fp_sims)
        trav_norm = norm(trav_sims)
        
        # Calculate Consensus Score (Average of all normalized scores)
        consensus_scores = (svd_norm + gnn_norm + n2v_norm + np_norm + kg_norm + fp_norm + trav_norm) / 7.0
        
        sorted_indices = np.argsort(consensus_scores)[::-1]
        
        recommendations = []
        for rank, idx in enumerate(sorted_indices[:top_n], start=1):
            drug_name = drugs[idx]
            is_direct = gene_name in self.data["drug_targets"].get(drug_name, [])
            indications = self.data["indications"].get(drug_name, ["N/A"])
            
            # Count how many methods placed this drug in their individual top 20%
            threshold = 0.8
            methods_agreed = sum([
                svd_norm[idx] >= threshold,
                gnn_norm[idx] >= threshold,
                n2v_norm[idx] >= threshold,
                np_norm[idx] >= threshold,
                kg_norm[idx] >= threshold,
                fp_norm[idx] >= threshold,
                trav_norm[idx] >= threshold
            ])
            
            recommendations.append({
                "rank": rank,
                "drug": drug_name,
                "consensus_score": float(consensus_scores[idx]),
                "methods_agreed": int(methods_agreed),
                "type": "Direct Target" if is_direct else "Repurposed (Indirect)",
                "indications": ", ".join(indications),
                "scores": {
                    "SVD": float(svd_norm[idx]),
                    "GNN": float(gnn_norm[idx]),
                    "Node2Vec": float(n2v_norm[idx]),
                    "NetProp": float(np_norm[idx]),
                    "TransE": float(kg_norm[idx]),
                    "Fingerprint": float(fp_norm[idx]),
                    "Traversal": float(trav_norm[idx])
                }
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

    def recommend_for_disease(self, disease_name, method="gnn", top_n=10):
        """
        Given a disease name, recommends the top candidate genes and drugs.
        Uses the disease_genes mapping to find genes associated with the disease,
        then finds drugs that target those genes or are topologically near them.
        """
        self.load_models()
        
        diseases = self.data.get("diseases", [])
        disease_genes = self.data.get("disease_genes", {})
        
        if disease_name not in diseases:
            raise ValueError(f"Disease '{disease_name}' not found in database.")
            
        model_data = self.gnn_data if method == "gnn" else self.svd_data
        genes = model_data["genes"]
        drugs = model_data["drugs"]
        gene_embs = model_data["gene_embeddings"]
        drug_embs = model_data["drug_embeddings"]
        
        # Get genes associated with disease
        associated_genes = disease_genes.get(disease_name, [])
        if not associated_genes:
            return {"genes": [], "drugs": []}
            
        # Get embeddings for these associated genes
        assoc_gene_indices = [genes.index(g) for g in associated_genes if g in genes]
        if not assoc_gene_indices:
            return {"genes": [], "drugs": []}
            
        # Create a "disease profile" by averaging the embeddings of its associated genes
        disease_profile = np.mean(gene_embs[assoc_gene_indices], axis=0)
        
        # Find top similar genes to this profile (expanding the genetic signature)
        gene_sims = self.compute_cosine_similarity(disease_profile, gene_embs)
        top_gene_indices = np.argsort(gene_sims)[::-1][:top_n]
        
        gene_recs = []
        for idx in top_gene_indices:
            g_name = genes[idx]
            is_known = g_name in associated_genes
            gene_recs.append({
                "gene": g_name,
                "score": float(gene_sims[idx]),
                "type": "Known Marker" if is_known else "Predicted Marker"
            })
            
        # Find top similar drugs to this profile
        drug_sims = self.compute_cosine_similarity(disease_profile, drug_embs)
        top_drug_indices = np.argsort(drug_sims)[::-1][:top_n]
        
        drug_recs = []
        for rank, idx in enumerate(top_drug_indices, start=1):
            d_name = drugs[idx]
            indications = self.data["indications"].get(d_name, [])
            is_indicated = disease_name in indications
            drug_recs.append({
                "rank": rank,
                "drug": d_name,
                "score": float(drug_sims[idx]),
                "type": "Approved Indication" if is_indicated else "Repurposed Candidate",
                "indications": ", ".join(indications) if indications else "N/A"
            })
            
        return {
            "disease": disease_name,
            "genes": gene_recs,
            "drugs": drug_recs
        }

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
            
            # 1. Add direct targets
            targets = self.data["drug_targets"].get(query_name, [])
            for t in targets[:4]:
                if t in genes:
                    add_node(t, t, "gene")
                    links.append({"source": query_name, "target": t, "value": 1.5, "type": "direct"})
                    
            # 2. Add top recommended similar drugs
            recs = self.recommend_similar_drugs(query_name, method=method, top_n=top_n)
            for rec in recs:
                d = rec["drug"]
                add_node(d, d, "drug")
                links.append({"source": query_name, "target": d, "value": rec["score"], "type": "similar_drug"})
                
            # 3. Add disease indications
            indications = self.data["indications"].get(query_name, [])
            for ind in indications:
                add_node(ind, ind, "disease")
                links.append({"source": query_name, "target": ind, "value": 2.0, "type": "indication"})
                
        elif query_name in self.data.get("diseases", []):
            # Query is a Disease
            add_node(query_name, query_name, "disease")
            
            disease_data = self.recommend_for_disease(query_name, method=method, top_n=top_n)
            
            # Add top genes
            for rec in disease_data["genes"][:top_n]:
                g = rec["gene"]
                add_node(g, g, "gene")
                links.append({"source": query_name, "target": g, "value": rec["score"] * 1.5, "type": "disease_gene"})
                
            # Add top drugs
            for rec in disease_data["drugs"][:top_n]:
                d = rec["drug"]
                add_node(d, d, "drug")
                links.append({"source": query_name, "target": d, "value": rec["score"] * 1.5, "type": "disease_drug"})
        else:
            raise ValueError(f"Query node '{query_name}' not found in database.")
            
        return {"nodes": nodes, "links": links}

if __name__ == "__main__":
    recommender = BioRecommender()
    recommender.load_models()
    print("GNN recommendations for TP53:")
    print(recommender.recommend_drugs_for_gene("TP53", "gnn", 3))

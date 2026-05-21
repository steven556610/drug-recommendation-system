import os
import pickle
import numpy as np
from utils import get_logger

logger = get_logger("Fingerprint")

class FingerprintModel:
    """
    Lightweight Chemical Fingerprint Model inspired by DeepPurpose.
    Computes Tanimoto similarity between mock Morgan Fingerprints.
    """
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, "data", "processed", "sppm.pkl")
        self.model_path = os.path.join(project_root, "models", "fingerprint_sim.pkl")
        
    def train(self):
        logger.info("Initializing Chemical Fingerprint similarity computation...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
            
        drugs = data["drugs"]
        fingerprints = data.get("drug_fingerprints", {})
        
        num_drugs = len(drugs)
        sim_matrix = np.zeros((num_drugs, num_drugs), dtype=np.float32)
        
        logger.info(f"Computing Tanimoto similarity for {num_drugs} drugs...")
        
        # Tanimoto coefficient = (A intersect B) / (A union B)
        for i in range(num_drugs):
            fp1 = np.array(fingerprints.get(drugs[i], [0]*256))
            for j in range(i, num_drugs):
                if i == j:
                    sim_matrix[i, j] = 1.0
                    continue
                fp2 = np.array(fingerprints.get(drugs[j], [0]*256))
                intersection = np.sum(np.bitwise_and(fp1, fp2))
                union = np.sum(np.bitwise_or(fp1, fp2))
                sim = intersection / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
                
        # To recommend drugs for a *gene*, we map drug-drug similarity back to gene targets
        # Score(Drug, Gene) = Max_{KnownDrug for Gene} (Sim(Drug, KnownDrug))
        drug_targets = data["drug_targets"]
        genes = data["genes"]
        gene_to_idx = {g: k for k, g in enumerate(genes)}
        
        drug_gene_scores = np.zeros((num_drugs, len(genes)), dtype=np.float32)
        
        for g_idx, gene in enumerate(genes):
            # Find all drugs that target this gene
            targeting_drugs = [d_idx for d_idx, d in enumerate(drugs) if gene in drug_targets.get(d, [])]
            if not targeting_drugs:
                continue
                
            for d_idx in range(num_drugs):
                # Drug's score for this gene is its max similarity to any drug that targets this gene
                max_sim = np.max(sim_matrix[d_idx, targeting_drugs])
                drug_gene_scores[d_idx, g_idx] = max_sim
                
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "sim_matrix": sim_matrix,
                "drug_gene_scores": drug_gene_scores
            }, f)
            
        logger.info(f"Fingerprint similarity training complete. Saved to {self.model_path}")

if __name__ == "__main__":
    model = FingerprintModel()
    model.train()

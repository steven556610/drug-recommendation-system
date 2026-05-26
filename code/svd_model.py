import os
import pickle
import numpy as np
from scipy.sparse.linalg import svds
from utils import get_logger, get_project_root

logger = get_logger("SVDModel")

class SVDRecommender:
    def __init__(self, embedding_dim=16):
        self.root = get_project_root()
        self.sppm_path = os.path.join(self.root, "data", "processed", "sppm.pkl")
        self.model_dir = os.path.join(self.root, "models")
        self.embeddings_path = os.path.join(self.model_dir, "svd_embeddings.pkl")
        
        self.embedding_dim = embedding_dim
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.drugs = []
        self.genes = []
        self.drug_embeddings = None
        self.gene_embeddings = None

    def train(self):
        """
        Loads the computed SPPM, performs Singular Value Decomposition (SVD),
        and extracts low-dimensional latent embeddings for both drugs and genes/proteins.
        """
        logger.info("Initializing SVD training process...")
        
        if not os.path.exists(self.sppm_path):
            raise FileNotFoundError(f"SPPM not found at {self.sppm_path}. Please run data_pipeline first.")
            
        with open(self.sppm_path, "rb") as f:
            data = pickle.load(f)
            
        self.drugs = data["drugs"]
        self.genes = data["genes"]
        sppm = data["sppm"]
        
        # Dimensions check
        num_drugs, num_genes = sppm.shape
        k = min(self.embedding_dim, num_drugs - 1, num_genes - 1)
        logger.info(f"Factoring matrix {num_drugs}x{num_genes} with target rank k={k}")
        
        # Perform Singular Value Decomposition (Truncated SVD)
        # Using svds for sparse/dense efficient computation
        U, Sigma, Vt = svds(sppm.astype(np.float64), k=k)
        
        # SVD returns columns sorted by singular values. Reversing for descending order.
        U = U[:, ::-1]
        Sigma = Sigma[::-1]
        Vt = Vt[::-1, :]
        
        # Scale embeddings by square root of Singular Values
        sqrt_Sigma = np.diag(np.sqrt(Sigma))
        self.drug_embeddings = np.matmul(U, sqrt_Sigma)
        self.gene_embeddings = np.matmul(Vt.T, sqrt_Sigma)
        
        # Save embeddings
        output = {
            "drugs": self.drugs,
            "genes": self.genes,
            "drug_embeddings": self.drug_embeddings,
            "gene_embeddings": self.gene_embeddings,
            "embedding_dim": k
        }
        
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(output, f)
            
        logger.info(f"SVD training complete. Embeddings saved to {self.embeddings_path}")
        logger.info(f"Embeddings shapes - Drugs: {self.drug_embeddings.shape}, Genes: {self.gene_embeddings.shape}")
        
        # MLflow Integration
        try:
            import mlflow
            logger.info("Logging SVD training to MLflow...")
            
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            mlflow.set_experiment("biorec-repurposing")
            with mlflow.start_run(run_name="SVD_Training"):
                mlflow.log_params({
                    "model_type": "SVD",
                    "embedding_dim": self.embedding_dim,
                    "drugs_count": len(self.drugs),
                    "genes_count": len(self.genes),
                    "matrix_rows": sppm.shape[0],
                    "matrix_cols": sppm.shape[1]
                })
                mlflow.log_metrics({
                    "singular_value_mean": float(np.mean(Sigma)),
                    "singular_value_max": float(np.max(Sigma)),
                    "singular_value_min": float(np.min(Sigma))
                })
                mlflow.log_artifact(self.embeddings_path)
        except Exception as e:
            logger.warning(f"Failed to log SVD training to MLflow (this is fine if mlflow is offline): {e}")

        return output

    def load_embeddings(self):
        """
        Loads pre-trained SVD embeddings.
        """
        if not os.path.exists(self.embeddings_path):
            self.train()
            
        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)
            
        self.drugs = data["drugs"]
        self.genes = data["genes"]
        self.drug_embeddings = data["drug_embeddings"]
        self.gene_embeddings = data["gene_embeddings"]
        
        return data

if __name__ == "__main__":
    recommender = SVDRecommender()
    recommender.train()

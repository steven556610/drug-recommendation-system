import os
import json
import pickle
import random
import numpy as np
import torch
from scipy.sparse.linalg import svds
from utils import get_logger, get_project_root
from svd_model import SVDRecommender
from gnn_model import GNNTrainer, GraphAutoencoder

logger = get_logger("ValidationEngine")

class ModelValidator:
    def __init__(self):
        self.root = get_project_root()
        self.sppm_path = os.path.join(self.root, "data", "processed", "sppm.pkl")
        self.metrics_path = os.path.join(self.root, "models", "validation_metrics.json")
        
        self.drugs = []
        self.genes = []
        self.sppm = None
        self.ppi_graph = None

    def load_data(self):
        with open(self.sppm_path, "rb") as f:
            data = pickle.load(f)
        self.drugs = data["drugs"]
        self.genes = data["genes"]
        self.sppm = data["sppm"]
        self.ppi_graph = data["ppi_graph"]

    @staticmethod
    def calculate_metrics(y_true, y_scores):
        """
        Pure NumPy implementation of AUROC and AUPR (Average Precision).
        Guarantees 100% environment compatibility with no scikit-learn dependency.
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Sort scores in descending order
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_scores_sorted = y_scores[desc_score_indices]
        
        # 1. Compute AUROC
        # True Positives / False Positives cumulative sums
        tp_cumsum = np.cumsum(y_true_sorted)
        fp_cumsum = np.cumsum(1 - y_true_sorted)
        
        num_pos = np.sum(y_true)
        num_neg = len(y_true) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            return 0.5, 0.5, [0.0], [0.0], [0.0], [0.0]
            
        tpr = tp_cumsum / num_pos
        fpr = fp_cumsum / num_neg
        
        # Add boundary values [0,0] to curves
        tpr = np.concatenate(([0.0], tpr))
        fpr = np.concatenate(([0.0], fpr))
        
        # Integrate using Trapezoidal rule
        auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
        
        # 2. Compute AUPR (Average Precision)
        precisions = []
        recalls = []
        
        # Precision = TP / (TP + FP)
        # Recall = TP / total_pos
        for i in range(1, len(y_true_sorted) + 1):
            tp = np.sum(y_true_sorted[:i])
            fp = i - tp
            
            p = tp / i
            r = tp / num_pos
            precisions.append(p)
            recalls.append(r)
            
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Area under PR curve using trapezoidal rule or average precision formula
        aupr = float(np.sum(precisions * y_true_sorted) / num_pos)
        
        # Thin out curves to 50 points to optimize JSON payload size for frontend Chart.js
        indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        fpr_thinned = fpr[indices].tolist()
        tpr_thinned = tpr[indices].tolist()
        
        # For PR curve
        pr_indices = np.linspace(0, len(recalls) - 1, 50, dtype=int)
        recalls_thinned = [0.0] + recalls[pr_indices].tolist()
        precisions_thinned = [1.0] + precisions[pr_indices].tolist()
        
        return auroc, aupr, fpr_thinned, tpr_thinned, recalls_thinned, precisions_thinned

    @staticmethod
    def calculate_recall_at_k(y_true, y_scores, k_vals=[10, 50]):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        
        total_pos = np.sum(y_true)
        if total_pos == 0:
            return {k: 0.0 for k in k_vals}
            
        recalls = {}
        for k in k_vals:
            # Count positive instances in top-K predictions
            pos_in_k = np.sum(y_true_sorted[:k])
            recalls[k] = float(pos_in_k / total_pos)
        return recalls

    def run_cross_validation(self):
        """
        Executes a rigorous 5-Fold Cross-Validation for both SVD and GNN models.
        Splits direct drug-target interactions into training and testing sets,
        evaluating model ability to predict held-out interaction links.
        """
        logger.info("Executing 5-Fold Cross-Validation framework...")
        self.load_data()
        
        num_drugs, num_genes = self.sppm.shape
        
        # 1. Compile direct interaction list (positives) and non-interaction list (negatives)
        pos_pairs = []
        neg_pairs = []
        
        for d_idx in range(num_drugs):
            for g_idx in range(num_genes):
                if self.sppm[d_idx, g_idx] == 1.0: # Direct Target annotation
                    pos_pairs.append((d_idx, g_idx))
                elif self.sppm[d_idx, g_idx] == 0.0:
                    neg_pairs.append((d_idx, g_idx))
                    
        logger.info(f"Interaction sets compiled: Positives: {len(pos_pairs)}, Negatives: {len(neg_pairs)}")
        
        # Sample equal negatives for balanced binary classification evaluation
        random.seed(42)
        np.random.seed(42)
        neg_pairs = random.sample(neg_pairs, len(pos_pairs))
        
        # Combine and shuffle
        all_pairs = np.array(pos_pairs + neg_pairs)
        labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
        
        shuffled_indices = np.random.permutation(len(all_pairs))
        all_pairs = all_pairs[shuffled_indices]
        labels = labels[shuffled_indices]
        
        # Split into 5 folds
        num_folds = 5
        fold_size = len(all_pairs) // num_folds
        
        svd_y_true, svd_y_scores = [], []
        gnn_y_true, gnn_y_scores = [], []
        
        for fold in range(num_folds):
            logger.info(f"Evaluating Fold {fold+1}/{num_folds}...")
            
            # Divide train and test indexes
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(all_pairs)
            
            test_pairs = all_pairs[test_start:test_end]
            test_labels = labels[test_start:test_end]
            
            # Construct a validation SPPM with test interactions hidden (set to 0)
            val_sppm = self.sppm.copy()
            for (d_idx, g_idx), label in zip(test_pairs, test_labels):
                if label == 1:
                    val_sppm[d_idx, g_idx] = 0.0 # Hide target link
                    
            # ----------------------------------------------------
            # A. Evaluate SVD Fold
            # ----------------------------------------------------
            # Truncated SVD on the masked validation SPPM
            k = min(16, num_drugs - 2, num_genes - 2)
            U, Sigma, Vt = svds(val_sppm.astype(np.float64), k=k)
            U = U[:, ::-1]
            Sigma = Sigma[::-1]
            Vt = Vt[::-1, :]
            
            sqrt_Sigma = np.diag(np.sqrt(Sigma))
            val_drug_emb = np.matmul(U, sqrt_Sigma)
            val_gene_emb = np.matmul(Vt.T, sqrt_Sigma)
            
            # Predict scores for test pairs using cosine similarity of SVD embeddings
            for (d_idx, g_idx), label in zip(test_pairs, test_labels):
                d_v = val_drug_emb[d_idx]
                g_v = val_gene_emb[g_idx]
                
                # Cosine Similarity
                norm_d = np.linalg.norm(d_v)
                norm_g = np.linalg.norm(g_v)
                if norm_d > 0 and norm_g > 0:
                    score = np.dot(d_v, g_v) / (norm_d * norm_g)
                else:
                    score = 0.0
                svd_y_true.append(label)
                svd_y_scores.append(score)
                
            # ----------------------------------------------------
            # B. Evaluate GNN Fold
            # ----------------------------------------------------
            # Instantiate GNN and train for a fast cross-validation cycle (30 epochs)
            trainer = GNNTrainer(epochs=30, lr=0.01)
            normalized_adj, target_adj = trainer.construct_unified_adjacency(
                val_sppm, self.ppi_graph, self.genes
            )
            
            # Set up Graph Autoencoder
            model = GraphAutoencoder(
                num_nodes=trainer.num_nodes,
                input_dim=32,
                hidden_dim=32,
                embedding_dim=16
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            
            # Short training loop
            model.train()
            for epoch in range(30):
                optimizer.zero_grad()
                z, adj_pred = model(normalized_adj)
                loss = criterion(adj_pred, target_adj)
                loss.backward()
                optimizer.step()
                
            # Predict scores for test pairs
            model.eval()
            with torch.no_grad():
                z = model.encode(normalized_adj).numpy()
                
            val_gnn_drug_emb = z[0:num_drugs]
            val_gnn_gene_emb = z[num_drugs:trainer.num_nodes]
            
            for (d_idx, g_idx), label in zip(test_pairs, test_labels):
                d_v = val_gnn_drug_emb[d_idx]
                g_v = val_gnn_gene_emb[g_idx]
                
                norm_d = np.linalg.norm(d_v)
                norm_g = np.linalg.norm(g_v)
                if norm_d > 0 and norm_g > 0:
                    score = np.dot(d_v, g_v) / (norm_d * norm_g)
                else:
                    score = 0.0
                gnn_y_true.append(label)
                gnn_y_scores.append(score)
                
        # 3. Calculate Overall Metrics & Curves across all folds
        logger.info("Computing final evaluation curves and performance metrics...")
        
        # SVD
        svd_auroc, svd_aupr, svd_fpr, svd_tpr, svd_rec, svd_prec = self.calculate_metrics(
            svd_y_true, svd_y_scores
        )
        svd_recalls = self.calculate_recall_at_k(svd_y_true, svd_y_scores, k_vals=[10, 50])
        
        # GNN
        gnn_auroc, gnn_aupr, gnn_fpr, gnn_tpr, gnn_rec, gnn_prec = self.calculate_metrics(
            gnn_y_true, gnn_y_scores
        )
        gnn_recalls = self.calculate_recall_at_k(gnn_y_true, gnn_y_scores, k_vals=[10, 50])
        
        # Save metrics to JSON for frontend presentation
        metrics = {
            "svd": {
                "auroc": float(svd_auroc),
                "aupr": float(svd_aupr),
                "recall_10": float(svd_recalls[10]),
                "recall_50": float(svd_recalls[50]),
                "fpr": svd_fpr,
                "tpr": svd_tpr,
                "recall": svd_rec,
                "precision": svd_prec
            },
            "gnn": {
                "auroc": float(gnn_auroc),
                "aupr": float(gnn_aupr),
                "recall_10": float(gnn_recalls[10]),
                "recall_50": float(gnn_recalls[50]),
                "fpr": gnn_fpr,
                "tpr": gnn_tpr,
                "recall": gnn_rec,
                "precision": gnn_prec
            }
        }
        
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Cross-Validation complete! Saved results to {self.metrics_path}")
        logger.info(f"--- SVD --- AUROC: {svd_auroc:.4f} | AUPR: {svd_aupr:.4f} | Recall@10: {svd_recalls[10]:.4f}")
        logger.info(f"--- GNN --- AUROC: {gnn_auroc:.4f} | AUPR: {gnn_aupr:.4f} | Recall@10: {gnn_recalls[10]:.4f}")
        
        return metrics

if __name__ == "__main__":
    validator = ModelValidator()
    validator.run_cross_validation()

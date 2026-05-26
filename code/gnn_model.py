import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_logger, get_project_root

logger = get_logger("GNNModel")

class GCNLayer(nn.Module):
    """
    Custom Graph Convolutional Network (GCN) layer implemented in pure PyTorch.
    Avoids compilation dependencies associated with external GNN frameworks on Windows.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot (Xavier) initialization
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        x: Node feature matrix (num_nodes x in_features)
        adj: Normalized symmetric adjacency matrix (num_nodes x num_nodes)
        """
        # H = adj * X * W + b
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GraphAutoencoder(nn.Module):
    """
    2-Layer Graph Convolutional Autoencoder (GAE) for network reconstruction and embedding.
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()
        
        # Learnable node feature embeddings to start with (since biological nodes are symbolic)
        self.node_features = nn.Parameter(torch.FloatTensor(num_nodes, input_dim))
        
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.reset_parameters(num_nodes, input_dim)

    def reset_parameters(self, num_nodes, input_dim):
        stdv = 1.0 / np.sqrt(input_dim)
        self.node_features.data.uniform_(-stdv, stdv)

    def encode(self, adj):
        h = self.node_features
        h = self.activation(self.gcn1(h, adj))
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        return h

    def decode(self, z):
        # Predict dot product similarity between all nodes
        # reconstructed_adj = sigmoid(Z * Z^T)
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred

    def forward(self, adj):
        z = self.encode(adj)
        adj_pred = self.decode(z)
        return z, adj_pred


class GNNTrainer:
    def __init__(self, hidden_dim=32, embedding_dim=16, epochs=150, lr=0.01):
        self.root = get_project_root()
        self.sppm_path = os.path.join(self.root, "data", "processed", "sppm.pkl")
        self.model_dir = os.path.join(self.root, "models")
        self.model_path = os.path.join(self.model_dir, "gnn_model.pt")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.drugs = []
        self.genes = []
        self.num_drugs = 0
        self.num_genes = 0
        self.num_nodes = 0
        
        self.drug_embeddings = None
        self.gene_embeddings = None

    def construct_unified_adjacency(self, sppm, ppi_graph, genes):
        """
        Builds a unified, normalized symmetric adjacency matrix containing:
        - PPI connections (Gene-Gene)
        - SPPM proximity scores (Drug-Gene weighted virtual edges)
        """
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        self.num_drugs, self.num_genes = sppm.shape
        self.num_nodes = self.num_drugs + self.num_genes
        
        # Initialize full raw adjacency matrix
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        # 1. Fill Drug-Gene block with SPPM scores
        # Row 0 to num_drugs-1 are Drugs, num_drugs to num_nodes-1 are Genes
        adj[0:self.num_drugs, self.num_drugs:self.num_nodes] = sppm
        adj[self.num_drugs:self.num_nodes, 0:self.num_drugs] = sppm.T
        
        # 2. Fill Gene-Gene block with PPI connections
        for gene, neighbors in ppi_graph.items():
            g_idx = gene_to_idx[gene] + self.num_drugs
            for nbr in neighbors:
                if nbr in gene_to_idx:
                    nbr_idx = gene_to_idx[nbr] + self.num_drugs
                    adj[g_idx, nbr_idx] = 1.0  # Unweighted STRING links
                    
        # 3. Add Self-loops for GCN message passing stability
        adj = adj + np.eye(self.num_nodes, dtype=np.float32)
        
        # 4. Normalized Adjacency: D^-1/2 * A * D^-1/2
        row_sums = np.sum(adj, axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1e-5
        d_inv_sqrt = np.power(row_sums, -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        normalized_adj = np.matmul(np.matmul(D_inv_sqrt, adj), D_inv_sqrt)
        return torch.FloatTensor(normalized_adj), torch.FloatTensor(adj)

    def train(self):
        """
        Constructs the graph and trains the Graph Autoencoder to learn joint representations.
        """
        logger.info("Initializing GNN training suite...")
        
        if not os.path.exists(self.sppm_path):
            raise FileNotFoundError(f"SPPM not found at {self.sppm_path}. Run data_pipeline first.")
            
        with open(self.sppm_path, "rb") as f:
            data = pickle.load(f)
            
        self.drugs = data["drugs"]
        self.genes = data["genes"]
        
        normalized_adj, target_adj = self.construct_unified_adjacency(
            data["sppm"], data["ppi_graph"], self.genes
        )
        
        # Instantiate GNN
        # Node features initially defined as 32-dim learnable representations
        model = GraphAutoencoder(
            num_nodes=self.num_nodes, 
            input_dim=32, 
            hidden_dim=self.hidden_dim, 
            embedding_dim=self.embedding_dim
        )
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        # We calculate MSE loss to reconstruct weighted edges (SPPM values + PPI connections)
        criterion = nn.MSELoss()
        
        # MLflow Integration
        self.active_run = None
        try:
            import mlflow
            logger.info("Initializing MLflow for GNN training...")
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("biorec-repurposing")
            self.active_run = mlflow.start_run(run_name="GNN_Training")
            mlflow.log_params({
                "model_type": "Weighted-GNN",
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "num_nodes": self.num_nodes,
                "num_drugs": self.num_drugs,
                "num_genes": self.num_genes
            })
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow for GNN: {e}")
        
        logger.info(f"Model Architecture:\n{model}")
        logger.info(f"Beginning training for {self.epochs} epochs...")
        
        model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            z, adj_pred = model(normalized_adj)
            
            # Reconstruction Loss
            loss = criterion(adj_pred, target_adj)
            loss.backward()
            optimizer.step()
            
            if epoch % 25 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{self.epochs} | Graph Reconstruction Loss: {loss.item():.5f}")
                try:
                    if self.active_run:
                        import mlflow
                        mlflow.log_metric("reconstruction_loss", loss.item(), step=epoch)
                except Exception:
                    pass
                
        # Extract embeddings
        model.eval()
        with torch.no_grad():
            final_embeddings = model.encode(normalized_adj).numpy()
            
        self.drug_embeddings = final_embeddings[0:self.num_drugs]
        self.gene_embeddings = final_embeddings[self.num_drugs:self.num_nodes]
        
        # Save model and embeddings
        output = {
            "drugs": self.drugs,
            "genes": self.genes,
            "drug_embeddings": self.drug_embeddings,
            "gene_embeddings": self.gene_embeddings,
            "model_state": model.state_dict()
        }
        
        # PyTorch save
        torch.save(output, self.model_path)
        
        logger.info(f"GNN training complete. Joint representations saved to {self.model_path}")
        logger.info(f"Embeddings shapes - Drugs: {self.drug_embeddings.shape}, Genes: {self.gene_embeddings.shape}")
        
        # Close MLflow run
        try:
            if self.active_run:
                import mlflow
                mlflow.log_artifact(self.model_path)
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to close MLflow run: {e}")
            
        return output

    def load_embeddings(self):
        """
        Loads pre-trained GNN joint embeddings.
        """
        if not os.path.exists(self.model_path):
            self.train()
            
        # Standard map_location to cpu for compatibility
        data = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False)
        self.drugs = data["drugs"]
        self.genes = data["genes"]
        self.drug_embeddings = data["drug_embeddings"]
        self.gene_embeddings = data["gene_embeddings"]
        
        return data

if __name__ == "__main__":
    trainer = GNNTrainer(epochs=10)
    trainer.train()

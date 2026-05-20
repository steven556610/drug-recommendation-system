import os
import json
import pickle
import random
import numpy as np
from utils import get_logger, get_project_root

logger = get_logger("DataPipeline")

class DataPipeline:
    def __init__(self):
        self.root = get_project_root()
        self.raw_dir = os.path.join(self.root, "data", "raw")
        self.processed_dir = os.path.join(self.root, "data", "processed")
        
        # Ensure folders exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.sppm_path = os.path.join(self.processed_dir, "sppm.pkl")
        self.demo_data_path = os.path.join(self.processed_dir, "demo_data.json")

    def generate_mock_data(self):
        """
        Generates realistic biological mock data representing:
        - A Protein-Protein Interaction (PPI) network of key human genes (like EGFR, TP53, TNF).
        - A set of approved drugs and their known protein targets (e.g. DrugBank-like links).
        - Known drug-disease indications for downstream validation.
        """
        logger.info("Generating biologically realistic mock datasets...")
        
        # 1. Selection of realistic human genes/proteins across major pathways
        cancer_genes = ["TP53", "EGFR", "BRCA1", "BRCA2", "AKT1", "MTOR", "KRAS", "HRAS", "ABL1", "KIT", "PDGFRA", "PDGFRB", "BRAF", "PTEN", "MDM2", "PIK3CA", "SRC", "ERBB2", "MET", "MYC"]
        immune_genes = ["TNF", "IL6", "IL1B", "IL2", "JAK1", "JAK2", "JAK3", "STAT3", "NFKB1", "CXCR4", "CCL2", "CD4", "CD8A", "CTLA4", "PDCD1", "IFNG", "TGFB1", "MAPK1", "MAPK3", "MAPK8"]
        metabolic_genes = ["INS", "INSR", "PPARG", "AMPK1", "SREBF1", "HMGCR", "LDLR", "APOB", "CPT1A", "FASN", "SIRT1", "LEP", "LEPR", "ADIPOQ", "GCK", "PKM", "G6PD", "PCK1", "FABP4", "CD36"]
        cardio_genes = ["ACE", "AGT", "AGTR1", "NOS3", "EDN1", "KCNH2", "SCN5A", "RYR2", "CACNA1C", "ADRB1", "ADRB2", "AVPR1A", "NPPA", "NPPB", "REN", "PLAT", "F2", "F10", "SERPINE1", "VWF"]
        
        all_genes = sorted(list(set(cancer_genes + immune_genes + metabolic_genes + cardio_genes)))
        num_genes = len(all_genes)
        
        # 2. Build a small-world PPI network (STRING style)
        # We want to connect genes within classes strongly, and add cross-connections
        ppi_edges = []
        
        # Helper to add edge
        def add_ppi(g1, g2):
            if g1 != g2 and (g1, g2) not in ppi_edges and (g2, g1) not in ppi_edges:
                ppi_edges.append((g1, g2))
                
        # Connect classes in a chain-like structure
        for gene_list in [cancer_genes, immune_genes, metabolic_genes, cardio_genes]:
            for i in range(len(gene_list)):
                # Connect sequentially
                add_ppi(gene_list[i], gene_list[(i+1)%len(gene_list)])
                # Add random local links
                for _ in range(2):
                    g2 = random.choice(gene_list)
                    add_ppi(gene_list[i], g2)
                    
        # Add bridges between classes
        classes = [cancer_genes, immune_genes, metabolic_genes, cardio_genes]
        for idx in range(len(classes)):
            next_idx = (idx + 1) % len(classes)
            for _ in range(5): # 5 cross-class links
                g1 = random.choice(classes[idx])
                g2 = random.choice(classes[next_idx])
                add_ppi(g1, g2)
                
        # 3. Create realistic Drugs and their Target Mappings (DrugBank style)
        drugs_dict = {
            "Imatinib": {"targets": ["ABL1", "KIT", "PDGFRB", "PDGFRA"], "diseases": ["Leukemia", "Gastrointestinal Stromal Tumor"]},
            "Gefitinib": {"targets": ["EGFR", "ERBB2"], "diseases": ["Lung Cancer", "Breast Cancer"]},
            "Trastuzumab": {"targets": ["ERBB2"], "diseases": ["Breast Cancer", "Gastric Cancer"]},
            "Aspirin": {"targets": ["PTGS1", "PTGS2", "TNF", "NFKB1"], "diseases": ["Inflammation", "Pain", "Thrombosis"]}, # ptgs1/2 mapped to inflammation nodes
            "Metformin": {"targets": ["AMPK1", "SIRT1", "PPARG"], "diseases": ["Type 2 Diabetes", "Obesity"]},
            "Atorvastatin": {"targets": ["HMGCR", "LDLR"], "diseases": ["Hypercholesterolemia", "Cardiovascular Disease"]},
            "Lisinopril": {"targets": ["ACE", "NOS3"], "diseases": ["Hypertension", "Heart Failure"]},
            "Everolimus": {"targets": ["MTOR", "AKT1"], "diseases": ["Renal Cancer", "Breast Cancer"]},
            "Captopril": {"targets": ["ACE"], "diseases": ["Hypertension"]},
            "Sorafenib": {"targets": ["BRAF", "KIT", "PDGFRB", "VEGFA"], "diseases": ["Renal Cancer", "Hepatocellular Carcinoma"]},
            "Infliximab": {"targets": ["TNF"], "diseases": ["Rheumatoid Arthritis", "Crohns Disease"]},
            "Tocilizumab": {"targets": ["IL6"], "diseases": ["Rheumatoid Arthritis", "COVID-19 Cytokine Storm"]},
            "Doxorubicin": {"targets": ["TP53", "BRCA1", "MYC"], "diseases": ["Lymphoma", "Breast Cancer", "Sarcoma"]},
            "Paclitaxel": {"targets": ["AKT1", "MAPK1", "TP53"], "diseases": ["Ovarian Cancer", "Breast Cancer", "Lung Cancer"]},
            "Valsartan": {"targets": ["AGTR1"], "diseases": ["Hypertension", "Heart Failure"]},
            "Propranolol": {"targets": ["ADRB1", "ADRB2"], "diseases": ["Hypertension", "Arrhythmia", "Anxiety"]},
            "Amlodipine": {"targets": ["CACNA1C"], "diseases": ["Hypertension", "Angina"]},
            "Warfarin": {"targets": ["F2", "F10"], "diseases": ["Thrombosis", "Atrial Fibrillation"]},
            "Insulin": {"targets": ["INSR", "INS"], "diseases": ["Diabetes mellitus"]},
            "Pioglitazone": {"targets": ["PPARG", "SREBF1"], "diseases": ["Type 2 Diabetes"]}
        }
        
        # Add random generic drugs to fill the database to 50 drugs
        generic_drug_names = [
            "BioMed-101", "OncoStop-A", "CardioGlow", "MetaboFix", "NeuroZepam",
            "Immunex-Beta", "ArthriCure", "LipidDown", "VasoFlow", "DiabeControl",
            "Zetaplat", "DuoCaine", "PressoGard", "ThromboNil", "StatGuard",
            "RespiClean", "GeneMend", "CellShield", "ProteoBlock", "Apoptin",
            "Mitocare", "KinaseOff", "CytokineBan", "SignalHalve", "AngioStop",
            "GlycoReg", "MyoRelax", "AntiThromb", "FibroBan", "OxiClear"
        ]
        
        for name in generic_drug_names:
            # Pick a pathway category
            target_class = random.choice([cancer_genes, immune_genes, metabolic_genes, cardio_genes])
            targets = random.sample(target_class, k=random.randint(1, 3))
            disease = random.choice(["Oncology", "Autoimmune", "Metabolic Syndrome", "Hypertension"])
            drugs_dict[name] = {
                "targets": targets,
                "diseases": [disease]
            }
            
        # Compile everything into a single JSON
        demo_data = {
            "genes": all_genes,
            "ppi_edges": ppi_edges,
            "drugs": drugs_dict
        }
        
        with open(self.demo_data_path, "w", encoding="utf-8") as f:
            json.dump(demo_data, f, indent=4)
            
        logger.info(f"Mock dataset generated successfully at {self.demo_data_path}")
        return demo_data

    def load_data(self):
        """
        Loads dataset. Checks if real STRING & DrugBank data files exist in `data/raw`.
        If not, automatically falls back to generating and loading the mock dataset.
        """
        if not os.path.exists(self.demo_data_path):
            self.generate_mock_data()
            
        with open(self.demo_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data

    def compute_sppm(self, alpha=0.5, max_hops=3):
        """
        Computes the Shortest-Path Proximity Matrix (SPPM) using BFS on the PPI network.
        Inspired by the methodology in Ceddia et al. (2020).
        """
        logger.info("Starting Shortest-Path Proximity Matrix (SPPM) computation...")
        
        data = self.load_data()
        genes = data["genes"]
        drugs = data["drugs"]
        ppi_edges = data["ppi_edges"]
        
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        num_genes = len(genes)
        
        # 1. Build adjacency list of the PPI network
        ppi_graph = {gene: set() for gene in genes}
        for g1, g2 in ppi_edges:
            # Ensure nodes are in our gene set
            if g1 in ppi_graph and g2 in ppi_graph:
                ppi_graph[g1].add(g2)
                ppi_graph[g2].add(g1)
                
        # 2. Compute SPPM
        drug_names = list(drugs.keys())
        num_drugs = len(drug_names)
        
        # Initialize dense similarity matrix S_prox
        sppm = np.zeros((num_drugs, num_genes), dtype=np.float32)
        
        for drug_idx, drug_name in enumerate(drug_names):
            targets = drugs[drug_name]["targets"]
            
            # Find valid targets present in our PPI graph
            valid_targets = [t for t in targets if t in ppi_graph]
            
            if not valid_targets:
                continue
                
            # Perform BFS from all target nodes simultaneously to find shortest paths to all genes
            distances = {t: 0 for t in valid_targets}
            queue = list(valid_targets)
            head = 0
            
            while head < len(queue):
                curr = queue[head]
                head += 1
                curr_dist = distances[curr]
                
                # Stop propagation if distance exceeds max_hops
                if curr_dist >= max_hops:
                    continue
                    
                for neighbor in ppi_graph[curr]:
                    if neighbor not in distances:
                        distances[neighbor] = curr_dist + 1
                        queue.append(neighbor)
                        
            # Apply decay function based on Dijkstra/BFS path distances
            for gene_name, dist in distances.items():
                gene_idx = gene_to_idx[gene_name]
                if dist == 0:
                    sppm[drug_idx, gene_idx] = 1.0  # Direct target
                else:
                    sppm[drug_idx, gene_idx] = alpha ** dist  # Indirect target proximity decay
                    
        # 3. Compile and save results
        output = {
            "drugs": drug_names,
            "genes": genes,
            "sppm": sppm,
            "ppi_graph": {g: list(neighbors) for g, neighbors in ppi_graph.items()},
            "drug_targets": {d: drugs[d]["targets"] for d in drug_names},
            "indications": {d: drugs[d]["diseases"] for d in drug_names}
        }
        
        with open(self.sppm_path, "wb") as f:
            pickle.dump(output, f)
            
        logger.info(f"SPPM calculated and saved to {self.sppm_path}")
        logger.info(f"Matrix Dimensions: {sppm.shape} (Drugs x Genes)")
        logger.info(f"Sparsity: {100.0 * (1.0 - np.count_nonzero(sppm) / sppm.size):.2f}% non-zero elements")
        
        return output

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.compute_sppm()

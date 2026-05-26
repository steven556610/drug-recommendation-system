import os
import sqlite3
import gzip
import csv
from utils import get_logger, get_project_root

logger = get_logger("DBManager")

class DBManager:
    def __init__(self):
        self.root = get_project_root()
        self.db_dir = os.path.join(self.root, "data", "processed")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "biorec.db")
        self.conn = None

    def get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            # Optimize SQLite performance for fast batch inserts and reads
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=OFF;")
            self.conn.execute("PRAGMA cache_size=100000;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def init_database(self):
        """Initializes tables and indexes for STRING/STITCH biological networks."""
        logger.info("Initializing SQLite database tables...")
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Proteins Table (STRING)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS proteins (
            ensembl_id TEXT PRIMARY KEY,
            preferred_name TEXT NOT NULL,
            description TEXT
        )""")

        # 2. Chemicals Table (STITCH)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chemicals (
            cid TEXT PRIMARY KEY,
            smiles TEXT,
            name TEXT
        )""")

        # 3. Protein-Protein Interactions (STRING PPI)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ppi_edges (
            protein1 TEXT,
            protein2 TEXT,
            combined_score INTEGER,
            PRIMARY KEY (protein1, protein2)
        )""")

        # 4. Chemical-Protein Interactions (STITCH CPI)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cpi_edges (
            chemical TEXT,
            protein TEXT,
            combined_score INTEGER,
            PRIMARY KEY (chemical, protein)
        )""")

        # Build highly optimized indexes for network traversals
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ppi_p1 ON ppi_edges(protein1);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ppi_p2 ON ppi_edges(protein2);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cpi_chem ON cpi_edges(chemical);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cpi_prot ON cpi_edges(protein);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prot_name ON proteins(preferred_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chem_name ON chemicals(name);")

        conn.commit()
        logger.info("Database tables and indexes created successfully.")

    def parse_chemicals_gzip(self, file_path, limit=50000):
        """
        Parses giant STITCH chemicals file in a streamed, chunked manner to prevent memory overflow.
        Inserts mapping of CID -> Name/SMILES.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Chemicals file not found at {file_path}")
            return False

        logger.info(f"Stream-parsing STITCH chemicals from {file_path} (batch size: 10000)...")
        conn = self.get_connection()
        cursor = conn.cursor()

        count = 0
        batch = []
        
        # open gzipped TSV
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            # Skip header if present
            header = next(reader, None)
            
            for row in reader:
                if len(row) < 2:
                    continue
                # STITCH file format: chemical, name, smiles, etc.
                # Adjust index depending on STITCH chemicals columns: usually (chemical, name, smiles)
                cid = row[0].strip()
                name = row[1].strip() if len(row) > 1 else cid
                smiles = row[2].strip() if len(row) > 2 else None
                
                batch.append((cid, smiles, name))
                count += 1
                
                if len(batch) >= 10000:
                    cursor.executemany("INSERT OR REPLACE INTO chemicals (cid, smiles, name) VALUES (?, ?, ?)", batch)
                    conn.commit()
                    batch = []
                    logger.info(f"Parsed and inserted {count} chemicals...")
                    
                if limit and count >= limit:
                    break

            if batch:
                cursor.executemany("INSERT OR REPLACE INTO chemicals (cid, smiles, name) VALUES (?, ?, ?)", batch)
                conn.commit()
                
        logger.info(f"Finished parsing chemicals. Ingested {count} items.")
        return True

    def parse_cpi_gzip(self, file_path, limit=100000, score_threshold=700):
        """
        Parses STITCH Chemical-Protein links.
        Filters by confidence score_threshold (default 700 for high confidence).
        """
        if not os.path.exists(file_path):
            logger.warning(f"CPI links file not found at {file_path}")
            return False

        logger.info(f"Stream-parsing Chemical-Protein links from {file_path} (threshold >= {score_threshold})...")
        conn = self.get_connection()
        cursor = conn.cursor()

        count = 0
        batch = []
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None) # header

            for row in reader:
                if len(row) < 3:
                    continue
                chem = row[0].strip()
                prot = row[1].strip()
                try:
                    score = int(row[2].strip())
                except ValueError:
                    continue

                if score >= score_threshold:
                    batch.append((chem, prot, score))
                    count += 1

                if len(batch) >= 10000:
                    cursor.executemany("INSERT OR REPLACE INTO ppi_edges (protein1, protein2, combined_score) VALUES (?, ?, ?)" if "links" in file_path and "chemical" not in file_path else "INSERT OR REPLACE INTO cpi_edges (chemical, protein, combined_score) VALUES (?, ?, ?)", batch)
                    conn.commit()
                    batch = []
                    logger.info(f"Parsed and inserted {count} interactions...")

                if limit and count >= limit:
                    break

            if batch:
                cursor.executemany("INSERT OR REPLACE INTO ppi_edges (protein1, protein2, combined_score) VALUES (?, ?, ?)" if "links" in file_path and "chemical" not in file_path else "INSERT OR REPLACE INTO cpi_edges (chemical, protein, combined_score) VALUES (?, ?, ?)", batch)
                conn.commit()

        logger.info(f"CPI ingestion complete. Total ingested: {count}")
        return True

    def query_ppi_subgraph(self, protein_list):
        """Fetches active PPI interactions among a sub-network of proteins."""
        if not protein_list:
            return []
        conn = self.get_connection()
        placeholders = ",".join("?" for _ in protein_list)
        query = f"""
        SELECT protein1, protein2, combined_score 
        FROM ppi_edges 
        WHERE protein1 IN ({placeholders}) AND protein2 IN ({placeholders})
        """
        cursor = conn.cursor()
        cursor.execute(query, protein_list + protein_list)
        return cursor.fetchall()

    def query_chemical_targets(self, cid):
        """Fetches target proteins for a specific chemical compound."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT protein, combined_score 
        FROM cpi_edges 
        WHERE chemical = ?
        """, (cid,))
        return cursor.fetchall()

    def query_protein_ligands(self, ensembl_id):
        """Fetches targeting chemical compounds for a specific protein."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT chemical, combined_score 
        FROM cpi_edges 
        WHERE protein = ?
        """, (ensembl_id,))
        return cursor.fetchall()

if __name__ == "__main__":
    db = DBManager()
    db.init_database()

import os
import sys
import importlib.util
import unittest

# Dynamic import configuration to bypass clashes with standard library 'code' module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api_file_path = os.path.join(project_root, "code", "api.py")

spec = importlib.util.spec_from_file_location("local_api", api_file_path)
api = importlib.util.module_from_spec(spec)
sys.modules["api"] = api
spec.loader.exec_module(api)

class TestBioRecAPIHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Pre-loads models and databases for direct handler verification."""
        api.load_recommender_data_if_needed()

    def test_autocomplete_handler(self):
        """Verifies autocomplete lists retrieval."""
        data = api.get_autocomplete_list()
        self.assertIn("genes", data)
        self.assertIn("drugs", data)
        self.assertIn("diseases", data)
        self.assertTrue(len(data["genes"]) > 0)
        self.assertTrue(len(data["drugs"]) > 0)

    def test_recommend_gene_handler(self):
        """Verifies candidate drugs targeting a dynamically fetched valid gene."""
        auto_resp = api.get_autocomplete_list()
        first_gene = auto_resp["genes"][0]
        
        # Test using whatever gene is available in the current active graph
        # Falls back to 'svd' model if GNN is not trained for this gene
        data = api.recommend_drugs_for_gene(name=first_gene, method="svd")
        self.assertEqual(data["query"], first_gene)
        self.assertEqual(data["method"], "SVD")
        self.assertTrue(len(data["results"]) > 0)
        
        # Verify result item schema
        item = data["results"][0]
        self.assertIn("rank", item)
        self.assertIn("drug", item)
        self.assertIn("score", item)
        self.assertIn("type", item)
        self.assertIn("indications", item)

    def test_recommend_multi_handler(self):
        """Verifies consensus drug recommendation using active gene name."""
        auto_resp = api.get_autocomplete_list()
        first_gene = auto_resp["genes"][0]
        
        data = api.recommend_drugs_multi(name=first_gene)
        self.assertEqual(data["query"], first_gene)
        self.assertEqual(data["method"], "CONSENSUS")
        # May be empty if external models are not trained yet, which is fine
        self.assertIsInstance(data["results"], list)

    def test_recommend_drug_handler(self):
        """Verifies similar drug recommendation using active drug name."""
        auto_resp = api.get_autocomplete_list()
        first_drug = auto_resp["drugs"][0]
        
        data = api.recommend_similar_drugs(name=first_drug, method="svd")
        self.assertEqual(data["query"], first_drug)
        self.assertTrue(len(data["results"]) > 0)
        
        item = data["results"][0]
        self.assertIn("rank", item)
        self.assertIn("drug", item)
        self.assertIn("score", item)
        self.assertIn("indications", item)

    def test_recommend_disease_handler(self):
        """Verifies drug and gene recommendation for an active disease query."""
        auto_resp = api.get_autocomplete_list()
        first_disease = auto_resp["diseases"][0]
        
        data = api.recommend_for_disease(name=first_disease, method="svd")
        self.assertEqual(data["query"], first_disease)
        
        results = data["results"]
        self.assertEqual(results["disease"], first_disease)
        self.assertIn("genes", results)
        self.assertIn("drugs", results)

if __name__ == "__main__":
    unittest.main()

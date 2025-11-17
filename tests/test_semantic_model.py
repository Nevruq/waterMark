import unittest
import sys
# Ergänzen des Zielverzeichnisses in sys.path
sys.path.insert(1, "/home/nev/Documents/Bachelor/RetrospectiveDetection/src")

import semantic_model as sm


class TestWatermarkLogitsManu(unittest.TestCase):

    def test_loadSemanticMode(self):
        semantic_model = sm.SemanticModel()
        semantic_model.load_semantic_probs()
        device = next(semantic_model.model.parameters()).device
        enc = semantic_model.tokenizer("This is a Test and more.", return_tensors="pt", max_length=1024)
        input_ids = enc.input_ids.to(device)  
        print(input_ids)
        assert True

if __name__ == "__main__":
    unittest.main()
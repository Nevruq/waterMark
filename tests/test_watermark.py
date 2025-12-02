import unittest
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM ,BertTokenizer, BertModel
import torch
# Ergänzen des Zielverzeichnisses in sys.path
sys.path.insert(1, "/home/nev/Documents/Bachelor/RetrospectiveDetection/src")

import watermark


class TestWatermark(unittest.TestCase):

    def test_watermarkSequence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = "super-secret-key"

        model_name = "gpt2"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        outputs = watermark.watermark_Sequence("This is a test Text.", model, tokenizer, "Random-key")
        

        assert True



if __name__ == "__main__":
    unittest.main()
import unittest
import sys
# Ergänzen des Zielverzeichnisses in sys.path
sys.path.insert(1, "/home/nev/Documents/Bachelor/RetrospectiveDetection/src")

import semantic_model as sm
import loadDataHugging as ldh


class TestWatermarkLogitsManu(unittest.TestCase):

    def test_loadSemanticMode(self):
        semantic_model = sm.SemanticModel()
        semantic_model.load_semantic_probs()
        device = next(semantic_model.model.parameters()).device
        enc = semantic_model.tokenizer("This is a Test and more.", return_tensors="pt", max_length=1024)
        input_ids = enc.input_ids.to(device)  
        #print(input_ids)
        #print(semantic_model.encode_text("This is a random Text."))
        semantic_model.build_mapping_list()
        #print(semantic_model.calc_cos_sim("this is a test and more", "The first man landed on the moon in 1969."))


        assert True # semantic_model.calc_cos_sim("hello this is the same", "hello this is the same") == 1
    
    def test_semanticCosSimilarity(self):
        # import org data
        df_org = ldh.load_datasetw()["train"]["human_answers"][:50]
        df_alter = ldh.load_jsonl("src/data/concat_set.jsonl")[:50]["context"]

        semantic_model = sm.SemanticModel()
        semantic_model.load_semantic_probs()
        device = next(semantic_model.model.parameters()).device 
        #print(input_ids)
        #print(semantic_model.encode_text("This is a random Text."))
        semantic_model.build_mapping_list()

        print(df_alter.shape)
        for org, alter in zip(df_org, df_alter):
            print(type(org), type(alter))
            #print(semantic_model.encode_text(org, alter))
        assert True



if __name__ == "__main__":
    unittest.main()
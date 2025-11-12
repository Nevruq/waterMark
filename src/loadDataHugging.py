import re
import random
import torch
import model as MyModel

from datasets import load_dataset

def load_datasetw():
    ds = load_dataset("Hello-SimpleAI/HC3", "finance")
    return ds

def count_double_WP(testData):
    """Counts amout of double Whitespaces in a Dataset
    Returns:
        Amount of Whitespaces
    """
    wp_count = 0
    for singleData in testData:
        split_tokens = re.findall(r"\S+|\s+", singleData[0])
        for token in split_tokens:
            if len(token) == 2 and token[0].isspace() and token[1].isspace():
                wp_count += 1
    return wp_count 

def load_semantic_model():
    # load the model in /Bachelor/RetrospectiveDetection/model/semantic_mapping_model.pth
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyModel.SemanticModel().to(device)

    state = torch.load("model/semantic_mapping_model.pth", map_location=device)

    # Falls sie als {"model_state_dict": ...} gespeichert wurde:
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # direktes state_dict
        model.load_state_dict(state)

    model.eval()
    return model

if __name__ == "__main__":
    load_semantic_model()
    #print(count_double_WP(load_datasetw()["train"]["human_answers"]) / 3933)
    
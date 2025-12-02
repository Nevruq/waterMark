import re
import random
import torch
import pandas as pd
import semantic_model as MyModel

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

def load_jsonl(path):
    df = pd.read_json(path)
    return df

if __name__ == "__main__":
    load_jsonl("src/data/concat_set.jsonl")
    #print(count_double_WP(load_datasetw()["train"]["human_answers"]) / 3933)
    
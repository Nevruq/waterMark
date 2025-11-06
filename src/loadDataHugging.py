import re
import random

from datasets import load_dataset

ds = load_dataset("Hello-SimpleAI/HC3", "finance")

# First 20 trainindata
testData = ds["train"]
randomData = random.sample(testData["human_answers"], 100)
print(len(randomData))

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
countt = count_double_WP(randomData)
print(countt)
print(len(testData) / countt)
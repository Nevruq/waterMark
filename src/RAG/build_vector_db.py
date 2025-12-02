# build_vector_db.py
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# 1) Embedding-Modell
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_raw_docs(path="data/raw_docs"):
    docs = []
    ids = []
    for fname in os.listdir(path):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
            docs.append(f.read())
            ids.append(fname)   # oder eine eigene ID
    return ids, docs

# 2) Watermark anwenden
#ids, docs = load_raw_docs()
wm_df = pd.read_json("src/data/concat_set.jsonl")

wm_docs = wm_df["context"].astype(str).tolist()


embeddings = embed_model.encode(
    wm_docs,
    convert_to_numpy=True,
    show_progress_bar=True
)

np.save("src/RAG/data_emb/embeddings.npy", embeddings)

"""
# 2) Texte + IDs speichern
with open("src/data/docs.json", "w", encoding="utf-8") as f:
    json.dump({"ids": ids, "docs": docs}, f, ensure_ascii=False, indent=2)
"""
print("Fertig: embeddings.npy und docs.json gespeichert.")

# load_rag_index.py
import json
import numpy as np
import faiss
import os

# 1) Embeddings laden
embeddings = np.load("src/RAG/data_emb/embeddings.npy")

# 2) Texte laden
with open("src/data/concat_set.jsonl", encoding="utf-8") as f:
    data = json.load(f)
docs = []
ids = []
for doc in data:
    docs.append(doc.get("context"))

ids = list(range(len(docs)))

# 3) FAISS Index bauen
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

print("Indexgröße:", index.ntotal)

print("Indexgröße:", index.ntotal)

# 4) Index + Docs + IDs speichern
os.makedirs("src/RAG/data_emb", exist_ok=True)

faiss.write_index(index, "src/RAG/data_emb/index.faiss")

with open("src/RAG/data_emb/docs.json", "w", encoding="utf-8") as f:
    json.dump({"ids": ids, "docs": docs}, f, ensure_ascii=False, indent=2)

print("Fertig: index.faiss und docs.json gespeichert.")

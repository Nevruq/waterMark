# rag_retriever.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Modelle & Daten laden
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("src/RAG/data_emb/index.faiss")

with open("src/RAG/data_emb/docs.json", encoding="utf-8") as f:
    data = json.load(f)
ids = data["ids"]
docs = data["docs"]

def retrieve(question: str, k: int = 3):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({
            "id": ids[idx],
            "doc": docs[idx],
            "distance": float(dist),
        })
    return results

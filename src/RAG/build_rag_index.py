# File to build the RAG index according to the provided data

# build_rag_index.py
from sentence_transformers import SentenceTransformer
import faiss, json, os  

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_watermarked_docs(path="data/watermarked_docs"):
    docs = []
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

docs = load_watermarked_docs()
emb = embed_model.encode(docs, convert_to_numpy=True)

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

faiss.write_index(index, "data/index.faiss")
with open("data/docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

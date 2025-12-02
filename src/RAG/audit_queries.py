# audit_queries.py
import json
from rag_server import answer

docs = json.load(open("data/docs.json", encoding="utf-8"))

def naive_question_from_doc(doc: str) -> str:
    # super simple: erste 1–2 Sätze nehmen und daraus eine Frage machen
    first_sentence = doc.split(".")[0]
    return f"Worum geht es in folgendem Inhalt im Detail: {first_sentence}?"

questions = [naive_question_from_doc(d) for d in docs]

responses = []
for q in questions:
    r = answer(q)
    responses.append({"q": q, "r": r})

with open("data/audit_responses.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)
# rag_answer.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from RAG.rag_retriever import retrieve

tokenizer = GPT2LMHeadModel.from_pretrained("gpt2").get_input_embeddings().weight.device
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def build_prompt(question: str, ctx_docs):
    ctx_blocks = []
    for i, hit in enumerate(ctx_docs):
        ctx_blocks.append(f"Dokument {i+1}:\n{hit['doc']}")
    ctx = "\n\n".join(ctx_blocks)
    prompt = (
        "Answers the following Questions exclusively with the Documents"
        "If something is not contained inside the Documents, explicetly mention that you do not know it."
        f"{ctx}\n\nFrage: {question}\nAntwort:"
    )
    return prompt

def answer(question: str, k: int = 3, max_new_tokens: int = 150) -> str:
    ctx_docs = retrieve(question, k=k)
    prompt = build_prompt(question, ctx_docs)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    out_ids = model.generate(   
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=5
                )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if "Antwort:" in text:
        text = text.split("Antwort:")[-1].strip()
    return text


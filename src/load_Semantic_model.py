from transformers import BertTokenizer, BertModel
import torch
import semantic_model as MyModel


# DEPRECATED !!!!!!!!!!!!!!!!!


def load_semantic_props():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BERT laden – muss derselbe sein wie beim Training!
    bert_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert = BertModel.from_pretrained(bert_name).to(device)
    bert.eval()

    # Dein SemanticModel + Gewichte laden
    semantic_model = MyModel.SemanticModel().to(device)
    state = torch.load("model/semantic_mapping_model.pth", map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        semantic_model.load_state_dict(state["model_state_dict"])
    else:
        semantic_model.load_state_dict(state)
    semantic_model.eval()

def encode_text(text: str):
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        out = bert(**enc)
        # Simple: Mean-Pooling über Tokens (wenn im Training auch so gemacht)
        cls_emb = out.last_hidden_state.mean(dim=1)  # [batch, 768]
        sem_emb = semantic_model(cls_emb)            # [batch, 384]
    return sem_emb

def calc_cos_sim(input1, input2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(encode_text(input1), encode_text(input2))

def calc_euclid_distance(input1, input2):


if __name__ == "__main__":
    emb = "This is a good sentence."
    emb2 = "This is a great sentence."
    emb3 = "Donald Trump is a stupid person."

    print(calc_cos_sim(emb, emb3))  
    print(calc_euclid_distance(emb, emb2))
    print(calc_euclid_distance(emb, emb3))

    print("test")
    
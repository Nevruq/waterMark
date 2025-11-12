from transformers import AutoTokenizer, AutoModelForCausalLM ,BertTokenizer, BertModel
import torch
import math
import hashlib
import pandas as pd
import csv
import random
import loadDataHugging as ldh
import load_Semantic_model as lsm
import binaryClassifierWatermark 
from scipy.stats import binom
from sklearn.metrics import confusion_matrix


def token_logprobs(text: str):
    # Tokenisieren
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    # Einmal forward: wir nutzen autoregressive Struktur
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits 

    # Für Token i nehmen wir logits an Position i-1
    # (erstes Token hat kein vorheriges -> überspringen)
    logprobs = []
    tokens = input_ids[0].tolist()

    # implement function that tracks the Sentence and when a end of a sentencee ist detected backtrack and check if its possible to change the end of the sentence 
    # to the most likely probability -> if the next most likely token is a dot too then swap the tokens.

    for i in range(1, len(tokens)):
        prev_logits = logits[0, i-1]                  # Logits für nächstes Token
        probs = torch.softmax(prev_logits, dim=-1)
        topk = torch.topk(probs, k=10)
        top_ids = topk.indices.tolist()
        top_probs = topk.values.tolist()

        """        print("\nTop-10 wahrscheinlichste nächste Tokens:")
        for tid, p in zip(top_ids, top_probs):
           
         print(f"{tokenizer.decode([tid]):10s}   {p:.4f}")
         """
        tok_id = tokens[i]
        lp = torch.log(probs[tok_id] + 1e-12).item()  # log P(x_i | x_<i)
        logprobs.append(lp)

    return tokens, logprobs, logits


def show_next_token_stats(text: str, top_k: int = 10):
    tokens, logprobs, logits = token_logprobs(text)
    # Calculate the entropie for each steps
    # Logits für das nächste Token nach dem letzten Input-Token
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)

    # Shannon-Entropie H(P) = -sum p log p
    log_probs = torch.log(probs + 1e-12)
    entropy_nat = -(probs * log_probs).sum().item()
    entropy_bits = entropy_nat / math.log(2)

    print(f"Text: {repr(text)}")
    print(f"Entropie (nat):  {entropy_nat:.4f}")
    print(f"Entropie (bits): {entropy_bits:.4f}\n")

    # Top-k wahrscheinlichste nächsten Tokens
    topk = torch.topk(probs, k=top_k)
    top_ids = topk.indices.tolist()
    top_ps = topk.values.tolist()

    print(f"Top-{top_k} nächste Tokens:")
    for tid, p in zip(top_ids, top_ps):
        tok = tokenizer.decode([tid])
        print(f"{repr(tok):12s}  P={p:.4f}")

def entropies_per_timestep(text: str, show_next_after_prompt: bool = True):
    # Tokenisieren
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits  # [1, seq_len, vocab_size]

    tokens = input_ids[0].tolist()
    decoded = tokenizer.convert_ids_to_tokens(tokens)

    entropies_bits = []

    # Für jedes Token x_i (i >= 1) Entropie von P(x_i | x_<i)
    # -> dafür verwenden wir logits an Position i-1
    print(tokens)
    for i in range(1, len(tokens)):
        if tokens[i] == 13: #Punkt
            print("erreicht punkt")
            print(step_logits)
        step_logits = logits[0, i-1]
        probs = torch.softmax(step_logits, dim=-1)

        log_probs = torch.log(probs + 1e-12)
        H_nat = -(probs * log_probs).sum().item()
        H_bits = H_nat / math.log(2)

        entropies_bits.append(H_bits)

    # Ausgabe: ein Eintrag pro "vorhergesagtem" Token
    print(f"Text: {repr(text)}\n")
    print(f"{'t':>3} {'token':>15} {'entropy(bits)':>15}")
    print("-" * 36)
    for t, (tok, H) in enumerate(zip(decoded[1:], entropies_bits), start=1):
        print(f"{t:3d} {tok:15s} {H:15.4f}")

    # Optional: Entropie für das nächste Token nach dem Prompt
    if show_next_after_prompt:
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        H_nat = -(probs * log_probs).sum().item()
        H_bits = H_nat / math.log(2)
        print("\nEntropie für das nächste Token nach dem gesamten Prompt:")
        print(f"H_next(bits) = {H_bits:.4f}")

    return entropies_bits

def score_function(gamma, score, sentiment_vector):
    return score + gamma * sentiment_vector

def replace_token_with_topk_alternative(text: str, gamma, k:int = 5):
    enc = tokenizer(text, return_tensors="pt", max_length=1024)
    input_ids = enc.input_ids.clone()          # [1, seq_len]
    seq_len = input_ids.size(1)

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits          # [1, seq_len, vocab_size]

    pos = seq_len - 2 #vorletzres token verändern

    step_logits = logits[0, pos-1]  # [vocab_size]
    probs = torch.softmax(step_logits, dim=-1)

    topk = torch.topk(probs, k=k)   
    top_ids = topk.indices.tolist()

    orig_id = input_ids[0, pos].item()
    alt_id = next(tid for tid in top_ids if tid != orig_id)

    # Logits für dieses Token kommen von der Position davor:
    # P(x_pos | x_<pos>) = logits[0, pos-1]
    # 1) Logit des ausgewählten Alternativ-Tokens anheben
    step_logits = logits[0, pos-1].clone()

    step_logits[alt_id] += gamma    

    # 2) Neue Verteilung aus den manipulierten Logits berechnen
    new_probs = torch.softmax(step_logits, dim=-1)

    # 3) Neues Token wählen (z.B. argmax oder sampeln)
    new_id = torch.multinomial(new_probs, num_samples=1).item()

    # 4) Token im Input ersetzen
    input_ids[0, pos] = new_id

    new_text = tokenizer.decode(input_ids[0])
    return new_text, orig_id, alt_id, top_ids

def sample_with_watermark(text, model, tokenizer, key, gamma=1.0, green_frac=0.5, max_new_tokens_per=0.2, entropie_nats_threshhold=7):

    tokens_changed = 0
    #pertuabte the last possible token vor dem punkt
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", max_length=1024)

    input_ids = enc.input_ids.to(device)  

    max_tokens_perb = int(len(input_ids) * max_new_tokens_per)
    green_set = get_green_set(50257, enc, key)
    
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits 

    # Initialize lists
    entropy_nats = []
    entropy_bits = []

    for t in range(1, logits.size(1)):  
        step_logits = logits[0, t-1]  # prediction for token t
        probs = torch.softmax(step_logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)

        # Shannon Entropy H = -Σ p log p
        H_nat = -(probs * log_probs).sum().item()
        H_bits = H_nat / math.log(2)

        entropy_nats.append(H_nat)
        entropy_bits.append(H_bits)

        if H_nat > entropie_nats_threshhold: # Wenn entropie höher als threshold
            k = 50
            topk = torch.topk(step_logits, k=k)
            top_ids = topk.indices.tolist()
            green_top = [tid for tid in top_ids if tid in green_set]
            #addiere gamma auf alle in dieser green top list

            boosted_logits = step_logits.clone()
            for tid in green_top:
                boosted_logits[tid] += gamma
            #sample ein neues Token aus dieser und ersetze es
            new_probs = torch.softmax(boosted_logits, dim=-1)
            new_token = torch.multinomial(new_probs, num_samples=1).item()

            old_input_ids = input_ids.clone()
            input_ids[0, t-1] = new_token
            
            decoded_inputs_old = tokenizer.decode(old_input_ids[0])
            decoded_inputs_new = tokenizer.decode(input_ids[0])

            cos_sim = lsm.calc_cos_sim(decoded_inputs_new, decoded_inputs_old)
            if not cos_sim > 0.9:
                input_ids = old_input_ids

    return input_ids
                
def get_green_set(vocab_size: int, context_ids, key: str, green_frac: float = 0.5):
    """
    Liefert ein Set von Token-IDs (Green List) für diese Position.
    green_frac = Anteil der Tokens, die grün sind (z.B. 0.5).
    """
    # Kontext in String gießen
    ctx_str = ",".join(map(str, context_ids))
    # cur seed ist key + | + die Liste selber encoded zu utf-8
    seed_input = (key + "|" + ctx_str).encode("utf-8")
    h = hashlib.sha256(seed_input).digest()
    seed = int.from_bytes(h, "big")

    rng = random.Random(seed)
    perm = list(range(vocab_size))
    rng.shuffle(perm)

    m = int(vocab_size * green_frac)
    green_ids = set(perm[:m])
    return green_ids

def is_green(token_id, context_ids, key, green_frac=0.5):
    ctx_str = ",".join(map(str, context_ids))
    s = f"{key}|{ctx_str}|{token_id}"
    h = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)
    threshold = int(green_frac * (2**32))
    return (h & 0xFFFFFFFF) < threshold

def detect_watermark(text, tokenizer, key, green_frac=0.5):
    # 1. Tokenisieren
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0].tolist()
    vocab_size = tokenizer.vocab_size

    # 2. Zähler initialisieren
    N = 0  # getestete Tokens
    K = 0  # Treffer im Green Set

    # 3. Für jede Position t das Green Set rekonstruieren
    for t in range(1, len(input_ids)):
        context = input_ids[:t]
        token_id = input_ids[t]
        
        if is_green(token_id, context, key, green_frac):
            K += 1
        N += 1

    # 4. Statistischer z-Score Test
    p = green_frac
    expected = p * N
    var = N * p * (1 - p)
    z = (K - expected) / math.sqrt(var) if var > 0 else 0.0
    p_value = 1 - binom.cdf(K - 1, N, p)

    return {
        "tokens_checked": N,
        "green_hits": K,
        "expected_hits": expected,
        "z_score": z,
        "p_value": p_value,
    }

def watermark_datafolder(model, tokenizer, key,amount_files, per_watermarked=0.3):
    """
    Watermarks a whole folder of files and creates new dataset
    """
    dataset = ldh.load_datasetw()["train"]["human_answers"]
    watermarked_csv = []
    id = 0
    for data in dataset[:amount_files]:
        device = next(model.parameters()).device
        enc = tokenizer(data, return_tensors="pt", max_length=1024)
        # checke ob Input weniger als 1024 tokens hat
        input_ids = enc.input_ids.to(device)[:1022]      
        text = tokenizer.decode(input_ids[0])
        if random.random() < per_watermarked:
            watermarked_text = sample_with_watermark(
                text,
                model,
                tokenizer,
                key,
            )
            text = tokenizer.decode(watermarked_text[0])
            # Lass die Detectorfunktion über den Text laufen
            res_detect = detect_watermark(text, tokenizer, key)
            watermarked_csv.append({
            "id": id,
            "is_watermarked_true": True,
            "tokens_length": len(text),
            "tokens_changed": "N.A",
            "tokens_checked": res_detect["tokens_checked"],
            "green_hits": res_detect["green_hits"],
            "expected_hits": res_detect["expected_hits"],
            "z_score": res_detect["z_score"],
            "p_value": res_detect["p_value"],
            "context": text
        })
        else:
            res_detect = detect_watermark(text, tokenizer, key)
            watermarked_csv.append({
                "id": id,
                "is_watermarked_true": False,
                "tokens_length": len(text),
                "tokens_changed": "N.A",
                "tokens_checked": res_detect["tokens_checked"],
                "green_hits": res_detect["green_hits"],
                "expected_hits": res_detect["expected_hits"],
                "z_score": res_detect["z_score"],
                "p_value": res_detect["p_value"],
                "context": text
        })
        print(id , " / ", amount_files, " are Done")
        id += 1
    # save csv in files
    with open(f"logit_watermarked_set.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=watermarked_csv[0].keys())
            writer.writeheader()
            writer.writerows(watermarked_csv)
    print("File fertig geschrieben")

# Beispiel: Nehme letzten token von Satz und passe ihn um einen Wert Delta an, wenn er höher als eine Entropie von z.B. 8 hat.

    

# Cosine Sim von 0.95 beschreibt in der Regel das gleiche

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = "super-secret-key"

    model_name = "gpt2"  # kleines Messmodell
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    #watermark_datafolder(model, tokenizer, key, 2000)
    
    BC = binaryClassifierWatermark.ClassifierLogitsManu(model, key, tokenizer)
    
    BC.train_classifier()
    




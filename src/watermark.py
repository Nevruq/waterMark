from transformers import AutoTokenizer, AutoModelForCausalLM ,BertTokenizer, BertModel
import torch
import math
import hashlib
import pandas as pd
import json
import os
import random
from tqdm import tqdm
import loadDataHugging as ldh
import binaryClassifierWatermark as bc
from scipy.stats import binom
from sklearn.metrics import confusion_matrix
import semantic_model

               

def watermark_per_Token(text, model, tokenizer, key,
                    gamma=3.0, threshold_bits=6.8, green_frac=0.5,
                    top_k=64, max_changes_per_seq=64, max_changes_per_seq_perc=0.5, dtype="fp16"):
    """
    Watermarkt jedes Token, welches eine Entropie höher als dem gegeben Entropie Threshold hat.
    """
    
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", max_length=1024)
    input_ids = enc.input_ids.to(device)  

    ids = input_ids.clone().to(device)

    if dtype == "fp16": 
        model.half()
        print("Model Type has been halfed.")

    model.eval()

    changes = 0
    seq_len = ids.size(1)

    with torch.inference_mode():
        # init cache mit erstem Token
        out = model(ids[:, :1], use_cache=True)
        past = out.past_key_values

        for i in range(1, seq_len):
            # 1) logits für Position i aus KV-Cache
            out = model(ids[:, i-1:i], past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[:, -1, :].squeeze(0)

            # 2) Entropie
            probs = torch.softmax(step_logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            H_bits = (-(probs * log_probs).sum().item()) / math.log(2)

            if H_bits <= threshold_bits or changes >= max_changes_per_seq or changes > max_changes_per_seq_perc * seq_len:
                # Überspringt token wenn: Entropie zu klein, Max Anzahl and changes pro Text erreicht, oder wenn Max Anzeil an Changes zu Sequence Länge erreicht ist
                # Teste mit Sequencelänge zu Tokenänderungen um zu vermeiden, dass Kurze Text hohen Anteil an Änderungen hat
                continue

            # 3) Green-Entscheidung nur auf Top-k (schneller & natürlich)
            topk = torch.topk(step_logits, k=top_k)
            cand_ids = topk.indices.tolist()
            prefix = ids[0, :i].tolist()
            green_top = [tid for tid in cand_ids if is_green(tid, prefix, key, green_frac)]
            if not green_top:
                continue

            # 4) logits boosten (nur green_top)
            boosted = step_logits.clone()
            boosted[green_top] += gamma

            new_probs = torch.softmax(boosted, dim=-1)
            new_id = torch.multinomial(new_probs, num_samples=1).item()

            # 5) Replace
            if new_id != ids[0, i].item():
                ids[0, i] = new_id
                changes += 1

    return ids, changes

def watermark_Sequence(text, model, tokenizer, key,
                    gamma=3.0, threshold_bits=6.8, green_frac=0.5,
                    top_k=64, max_changes_per_seq=64, max_changes_per_seq_perc=0.5, dtype="fp16", ignore_starting_seq=0.05):
    """
    Watermarkt nur in Sektionen in denen eine hohe Entropie herrscht. Verhindert wilkürliche Ersetzung von Tokens.
    Zudem berücksichte, dass der Anfang der Sequenz nicht verändert werden soll.
    """
    
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", max_length=1024)
    input_ids = enc.input_ids.to(device)  

    ids = input_ids.clone().to(device)

    if dtype == "fp16": 
        model.half()
        print("Model Type has been halfed.")

    model.eval()

    changes = 0
    seq_len = ids.size(1)
    H_bits_hist = []

    with torch.inference_mode():
        # init cache mit erstem Token
        out = model(ids[:, :1], use_cache=True)
        past = out.past_key_values

        for i in range(1, seq_len):
            # 1) logits für Position i aus KV-Cache
            out = model(ids[:, i-1:i], past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[:, -1, :].squeeze(0)

            # 2) Entropie
            probs = torch.softmax(step_logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            H_bits = (-(probs * log_probs).sum().item()) / math.log(2)
            H_bits_hist.append(H_bits)

            if i <= seq_len * ignore_starting_seq:
                # ignoriert die ersten ignore_starting percent um Semantic des Original Textes aufzubauen
                continue

            if H_bits <= threshold_bits or changes >= max_changes_per_seq or changes > max_changes_per_seq_perc * seq_len:
                # Überspringt token wenn: Entropie zu klein, Max Anzahl and changes pro Text erreicht, oder wenn Max Anzeil an Changes zu Sequence Länge erreicht ist
                # Teste mit Sequencelänge zu Tokenänderungen um zu vermeiden, dass Kurze Text hohen Anteil an Änderungen hat
                continue

            # Semantik Vector berechnen
            prev_ids = ids[0, :i] 
            sm = semantic_model().load_semantic_probs()
            sm.model               

            # 3) Green-Entscheidung nur auf Top-k (schneller & natürlich)
            topk = torch.topk(step_logits, k=top_k)
            cand_ids = topk.indices.tolist()
            prefix = ids[0, :i].tolist()
            green_top = [tid for tid in cand_ids if is_green(tid, prefix, key, green_frac)]
            if not green_top:
                continue

            # 4) logits boosten (nur green_top)
            boosted = step_logits.clone()
            boosted[green_top] += gamma

            new_probs = torch.softmax(boosted, dim=-1)
            new_id = torch.multinomial(new_probs, num_samples=1).item()

            # 5) Replace
            if new_id != ids[0, i].item():
                ids[0, i] = new_id
                changes += 1

    return ids, changes



def is_green(token_id, context_ids, key, green_frac=0.5):
    """Checkt determinischt Ob ein ein Token, wenn alle anderen Parameter gleich sein, 
        ob dieser in der Green/Red List ist."""

    # Erstellt unique Seed aus den ids des texts + key + dem momentaren Token, determinisch
    ctx_str = ",".join(map(str, context_ids))
    s = f"{key}|{ctx_str}|{token_id}"

    h = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)
    # sha256 gibt uns ein 256 bit hash
    threshold = int(green_frac * (2**32))
    # Der threshold ist in Green_frac=0.5 * 32Bits
    # Im letzten Schritt wird der hash, da er gleichverteilt ist durch Sha, auf 32 Bit abgeschnitten, und mit dem TH verglichen
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
        # context: bisherige Token_ids
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

def watermark_datafolder_json(model, tokenizer, key, dataset,
                              gamma, out_path="new_dataset.jsonl",
                              per_watermarked=0.4, flush_every=20, current_datatset=None):
    """
    Watermarks a dataset and saves results incrementally as JSONL.
    Each line = one sample.
    """

    # open file in append mode
    f = open(out_path, "a", encoding="utf-8")
    id_counter = 0
    #adjust ID counter to the last found datapoint
    if not current_datatset:
        #id started bei 0 -> + len(file)
        id_counter = len(pd.read_json(current_datatset, encoding="utf-8"))

    buffer = []

    try:
        for data in tqdm(dataset):
            text = data
            row = {}

            # --- 1. optionally watermark ---
            if random.random() < per_watermarked:
                watermarked_text_ids = watermark_per_Token(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    key=key,
                    gamma=gamma
                )
                text = tokenizer.decode(watermarked_text_ids[0][0])
                tokens_changed = watermarked_text_ids[1]
                is_wm = True
            else:
                tokens_changed = 0
                is_wm = False

            # --- 2. run detection ---
            res_detect = detect_watermark(text, tokenizer, key)

            # --- 3. build row dict ---
            row = {
                "id": id_counter,
                "is_watermarked_true": is_wm,
                "tokens_length": len(watermarked_text_ids[0]),
                "tokens_changed": tokens_changed,
                "tokens_checked": res_detect.get("tokens_checked", "N/A"),
                "green_hits": res_detect.get("green_hits", "N/A"),
                "expected_hits": res_detect.get("expected_hits", "N/A"),
                "z_score": res_detect.get("z_score", "N/A"),
                "p_value": res_detect.get("p_value", "N/A"),
                "context": text
            }

            buffer.append(row)
            id_counter += 1

            # --- 4. flush every N entries ---
            if len(buffer) >= flush_every:
                for entry in buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                buffer.clear()

        # write remaining buffer
        if buffer:
            for entry in buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        print(f" Finished writing {id_counter} samples to {out_path}")

    except KeyboardInterrupt:
        # ensure partial progress saved
        if buffer:
            for entry in buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print("\n Interrupted — progress saved up to this point.")
        raise

    finally:
        f.close()


def compare_semantic(dataset_1, dataset_2, model):
    s_model = semantic_model()

    




# Beispiel: Nehme letzten token von Satz und passe ihn um einen Wert Delta an, wenn er höher als eine Entropie von z.B. 8 hat.
# Cosine Sim von 0.95 beschreibt in der Regel das gleiche

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = "super-secret-key"

    model_name = "gpt2"  # kleines Messmodell
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    df = ldh.load_datasetw()["train"]["human_answers"][471:]

    watermark_datafolder_json(model, tokenizer, key, df, gamma=3)


    
    #BC = bc.ClassifierLogitsManu(model, key, tokenizer)
    
    #BC.train_classifier_jsonl("concat_set.jsonl")
    




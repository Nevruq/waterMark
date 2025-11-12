import re, math, hashlib, random
from typing import List, Tuple

# ---------- Hilfen ----------
def split_tokens_and_ws(text: str) -> List[str]:
    """Zerlegt in abwechselnd Nicht-Whitespace und Whitespace-Blöcke."""
    split_tokens = re.findall(r"\S+|\s+", text)
    return split_tokens

def word_gaps(parts: List[str]) -> List[int]:
    """
    Liefert die Indizes der Whitespace-Blöcke, die direkt zwischen zwei Wort-Token liegen.
    (parts[i] = token, parts[i+1] = ws, parts[i+2] = token)
    """
    gaps = []
    for i in range(0, len(parts)-2):
        if not parts[i].isspace() and parts[i+1].isspace() and not parts[i+2].isspace():
            gaps.append(i+1)  # Index des Whitespace-Blocks
    return gaps

def prng_indices(n_gaps: int, gamma: float, key: str, context:str , contextFunktion: bool = False) -> set:
    """Wählt deterministisch ~gamma*n_gaps Zielpositionen basierend auf key. Fur jeden Text werden die gleichen Stellen markiert"""
    
    #Funktion welche zum hashen benutzt werden soll
    seed = None
    # Deterministische Funktion anhand eines vorherigen Keys
    if contextFunktion:
        h = hashlib.sha256(key.encode(encoding="ascii")).digest()
        seed = int.from_bytes(h, "big") % (2**31-1)
    else:
        # Verändert seed = hash(key + context)
        seed = gen_seed_wContext(context, key)
    rnd = random.Random(seed)
    k = max(1, int(round(gamma * n_gaps)))
    return set(rnd.sample(range(n_gaps), k))  # Indizes 0..n_gaps-1


def gen_seed_wContext(context: str, key: str) -> int:
    """
    Erzeugt deterministischen Seed aus key + Textkontext.
    """
    h = hashlib.sha256()
    h.update(key.encode())
    h.update(context.encode('utf-8'))
    return int.from_bytes(h.digest(), "big") % (2**31-1) 

def count_double_WPs(text:str) -> int:
    """
    Counts how many Double Whitespaces a text contains.
    """
    wp_count = 0
    split_tokens = re.findall(r"\S+|\s+", text)
    for token in split_tokens:
        if len(token) == 2 and token[0].isspace() and token[1].isspace():
            wp_count += 1
    return wp_count 


# ---------- Encoder ----------
def encode_whitespace_watermark(text: str, key: str, gamma: float = 0.5, contextFunktion: bool = False) -> str:
    """
    Fügt an deterministisch ausgewählten Wort-Gaps doppelte Whitespaces ein.
    gamma = erwarteter Anteil markierter Gaps.
    """
    parts = split_tokens_and_ws(text)
    gaps = word_gaps(parts)                   # Liste von WS-Positions-Indizes in parts
    if not gaps:
        return text

    sel = prng_indices(len(gaps), gamma, key, text, contextFunktion) # Auswahl in Gap-Indexraum 0..len(gaps)-1
    for j, ws_idx in enumerate(gaps):
        if j in sel:
            # mind. zwei Leerzeichen (behalte Newlines bei, erweitere ansonsten auf "  ")
            ws = parts[ws_idx]
            if "\n" in ws or "\t" in ws:
                # füge zusätzlich ein Space an, damit Länge >= 2 bleibt sichtbar
                parts[ws_idx] = ws + " "
            else:
                parts[ws_idx] = "  "  # genau zwei Spaces
        else:
            parts[ws_idx] = " "       # Normalisierung auf genau ein Space

    return "".join(parts)

# ---------- Detector ----------

def detect_whitespace_watermark(text: str, key: str, gamma: float = 0.5, contextFunktion=False):
    """
    Zählt an den durch key+gamma bestimmten Ziel-Gaps, wie oft der WS-Block Länge >= 2 hat.
    Liefert z-Score, p-Wert (einseitig) und Rate.
    """
    parts = split_tokens_and_ws(text)
    gaps = word_gaps(parts)
    n = len(gaps)
    if n == 0:
        return {"n": 0, "k": 0, "z": 0.0, "p_value": 1.0, "rate": 0.0}

    sel = prng_indices(n, gamma, key, text, contextFunktion)

    k = 0
    for j, ws_idx in enumerate(gaps):
        if j in sel:
            ws = parts[ws_idx]
            # als "markiert" werten: mind. 2 Zeichen Whitespace
            if ws.isspace() and len(ws) >= 2:
                k += 1

    mean = n * gamma
    var = n * gamma * (1 - gamma)
    z = (k - mean) / math.sqrt(max(var, 1e-9))
    p = 0.5 * math.erfc(z / math.sqrt(2))  # einseitig

    return {"n": n, "k": k, "z": z, "p_value": p, "rate": k / n}

def detect_np_type(object_np, key):
    return detect_whitespace_watermark(object_np[1], key, gamma=0.5, contextFunktion=True)

# ---------- Mini-Demo ----------
if __name__ == "__main__":
    key = "super-secret"
    text = "This is aaa short example text, showing how a whitespace watermark works. And even further"

    wm = encode_whitespace_watermark(text, key, gamma=0.4, contextFunktion=True)
    wm = encode_whitespace_watermark(text, key, gamma=0.4, contextFunktion=False)
    print("WATERMARKED:\n", wm)

    nonAIExmp = "Dies  ist ein  Text der nicht KI generiert  ist und deswegen nur ein Whitespace-Block mit  2 leerzeichen hat."
    res = detect_whitespace_watermark(nonAIExmp, key, gamma=0.2)
    print("\nDETECTION:", res)
    # Daumenregel: z > 4 (oder p < 1e-4) => starkes Indiz für vorhandenes Watermark

    print("#Testing Count_DOUBLE_WP")
    print(count_double_WPs("This is a text with  2  wp."))

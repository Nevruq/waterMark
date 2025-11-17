import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from scipy.stats import binom
import math
import numpy as np
import matplotlib.pyplot as plt

import watermark as tlo


def ClassifierWhiteSpace():
    df = pd.read_csv("src/data/wp_results_gamma_0_3.csv")  

    df = df.dropna(subset=["is_watermarked_true", "amount_double_WP", "n"])
    # Removes all values expect, amount_WP, label, length of text
    print(df["n"])
    X = df[["amount_double_WP","n"]].astype(float).values
    y = df["is_watermarked_true"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Classifier
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """ 
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))


    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
    plt.xlabel("Amount of Double Whitespaces")
    plt.ylabel("Text Length (n)")
    plt.title("Predicted Watermarked (red) vs Non-Watermarked (blue)")
    plt.show()


class ClassifierLogitsManu():
    def __init__(self, model, key, tokenizer):
        self.model = model
        self.key = key
        self.tokenizer = tokenizer
        self.binClassifier = None
    
    def binary_classification(self, text, green_frac=0.5) -> bool:
        res = self.exam_text(text, green_frac=green_frac)
        z_score = res["z_score"]
        #T ≈ 3 → eher sensibel
        if z_score > 3.0 and N > 20:
            return True
        else:
            return False

        #T ≈ 4 → stark
        #T ≥ 5 → extrem konservativ (praktisch „sehr sicher“ watermarked)


    def exam_text(self, text, green_frac=0.5):
    # 1. Tokenisieren
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids[0].tolist()
        vocab_size = self.tokenizer.vocab_size

        # 2. Zähler initialisieren
        N = 0  # getestete Tokens
        K = 0  # Treffer im Green Set

        # 3. Für jede Position t das Green Set rekonstruieren
        for t in range(1, len(input_ids)):
            context = input_ids[:t]
            token_id = input_ids[t]
            green = tlo.get_green_set(vocab_size, context, self.key, green_frac)

            N += 1
            if token_id in green:
                K += 1

        # 4. Statistischer z-Score Test
        p = green_frac
        expected = p * N
        var = N * p * (1 - p)
        z = (K - expected) / math.sqrt(var) if var > 0 else 0.0
        p_value = 1 - binom.cdf(K - 1, N, p)

        print(z)

        return {
            "tokens_checked": N,
            "green_hits": K,
            "expected_hits": expected,
            "z_score": z,
            "p_value": p_value,
        }
    
    def train_classifier_jsonl(self, path, class_weight="balanced", save_to=None):
    # 1) JSONL laden
        df = pd.read_json(path)

        # 2) Label robust erstellen (True/False oder "true"/"false")
        if df["is_watermarked_true"].dtype == bool:
            df["label"] = df["is_watermarked_true"].astype(int)
        else:
            df["label"] = (
                df["is_watermarked_true"]
                .astype(str).str.strip().str.lower()
                .map({"true": 1, "false": 0})
            )

        # 3) benötigte Numerik-Spalten sicher in float konvertieren
        num_cols = ["green_hits", "tokens_checked", "z_score", "p_value", "tokens_length"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 4) Features ableiten
        df["hit_rate"] = df["green_hits"] / df["tokens_checked"].clip(lower=1)

        feature_cols = ["z_score", "p_value", "hit_rate", "tokens_length", "tokens_checked"]

        # 5) Zeilen mit fehlenden Werten verwerfen (nur was wir brauchen)
        df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)

        X = df[feature_cols]
        y = df["label"].astype(int)

        # 6) Split + Pipeline (Skalierung + LogReg)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=class_weight) 
        )

        clf.fit(X_train, y_train)

        # 7) Evaluation
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nReport:")
        print(classification_report(y_test, y_pred, digits=3))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))

        return clf, feature_cols



    def train_classifier_csv(self, path):  
        df = pd.read_csv(path)

        # 2. Label vorbereiten
        # falls 'True'/'False' als Strings drinstehen:


        df["label"] = df["is_watermarked_true"].astype(int)  # oder .map({"True":1, "False":0})

        # 3. Zusätzliche Features berechnen
        df["hit_rate"] = df["green_hits"] / df["tokens_checked"].clip(lower=1)

        # 4. Feature-Matrix X und Zielvariable y
        feature_cols = [
            "z_score",
            "p_value",
            "hit_rate",
            "tokens_length",
            "tokens_checked",
        ]

        X = df[feature_cols]
        y = df["label"]

        # 5. Train/Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 6. Pipeline: Skalierung + Logistic Regression
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nReport:")
        print(classification_report(y_test, y_pred, digits=3))

        print("ROC-AUC:", roc_auc_score(y_test, y_prob))



    def plot_informations(self, y_labels, y_pred):
        print("Confusion Matrix:")
        print(confusion_matrix(y_labels, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_labels, y_pred))
        

if __name__ == "__main__":
    pass
import loadDataHugging as ldh
import numpy as np
import random
import csv

import whiteSpaceDetector as wpd

length_data = 5000
key = "key"
gamma = [0.3]

# load the data 
ds = ldh.load_datasetw()["train"]["human_answers"]


def run_data_gen(gamma: list[float]):
    for single_gamma in gamma:
        dtype = np.dtype([
            ('id', np.int32),
            ('text', object),   
            ('label', np.bool_)
        ])
        id = 0
        data = []
        for single_data in ds:
            id += 1
            temp_data = (id, single_data[0], False)
            if random.random() < 0.3:
                # 30% of the data is gonna get watermarked
                watermarked_text = wpd.encode_whitespace_watermark(single_data[0],
                                                                                key, 
                                                                                gamma=0.02, 
                                                                                contextFunktion=True)
                
                temp_data = (id, watermarked_text, True)
            data.append(temp_data)

        data_np = np.array(data, dtype=dtype)
        # Run the detector and check how well it identifies Watermarked Data
        def detect_one(entry, key):
            _id, text, is_watermarked = entry
            res = wpd.detect_whitespace_watermark(text, key=key, gamma=0.5, contextFunktion=True)
            return res["p_value"], is_watermarked

        # Vectorized wrapper
        detect_all = np.vectorize(lambda e: detect_one(e, key))

        # Ergebnis
        z_scores = detect_all(data_np)

        results_csv = []

        for _id, text, is_wm in data_np:
            res = wpd.detect_whitespace_watermark(
                text,
                key=key,
                gamma=0.5,
                contextFunktion=True
            )
            results_csv.append({
                "id": _id,
                "is_watermarked_true": is_wm,
                "amount_double_WP": wpd.count_double_WPs(text),
                "z": res["z"],
                "p_value": res["p_value"],
                "rate": res["rate"],
                "n": res["n"],
                "k": res["k"],
                "context": text
            })
            if res["p_value"] <= 0.05:
                print(res)

        str_single_gamma = str(single_gamma).replace(".", "_")
        with open(f"data/wp_results_gamma_{str_single_gamma}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_csv[0].keys())
            writer.writeheader()
            writer.writerows(results_csv)


        print(f"Fertig! Für Gamme={str_single_gamma}Daten wurde ist das .csv geschrieben.")


if __name__ == "__main__":
    pass
# absa_end_to_end_all.py
import time
import pandas as pd
import os
from pyabsa import ATEPCCheckpointManager

start_time = time.time()

# Ordner
input_folder = "03_clean_reviews"
output_folder = "04.3.1_absa_files_end2end"
os.makedirs(output_folder, exist_ok=True)

# Lade ABSA-Modell (Aspect + Sentiment)
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint="multilingual",
    auto_device=True
)

def extract_aspects_from_review(review: str):
    result = aspect_extractor.extract_aspect(
        inference_source=[review],
        print_result=False
    )
    extracted = []
    if result and result[0]['aspect']:
        for asp, sent in zip(result[0]['aspect'], result[0]['sentiment']):
            extracted.append({"aspect": asp, "sentiment": sent})
    return extracted or [{"aspect": None, "sentiment": None}]

all_dfs = []

# Alle Dateien im Input-Ordner durchgehen
for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    input_path = os.path.join(input_folder, file)

    print(f"🔎 Verarbeite {file} ...")
    df = pd.read_csv(input_path)

    flat_data = []

    for _, row in df.iterrows():
        review = row["text_clean"]
        aspects = extract_aspects_from_review(review)

        for asp in aspects:
            new_row = row.to_dict()  # alle Originalspalten übernehmen
            new_row["aspect"] = asp["aspect"]
            new_row["sentiment"] = asp["sentiment"]
            flat_data.append(new_row)

    df_out = pd.DataFrame(flat_data)
    all_dfs.append(df_out)

# Alles zu einer großen Datei zusammenfassen
df_final = pd.concat(all_dfs, ignore_index=True)

output_path = os.path.join(output_folder, "end2end_all_aspects.csv")
df_final.to_csv(output_path, index=False, encoding="utf-8")

elapsed_time = time.time() - start_time
print(f"Alle ABSA-Ergebnisse gespeichert in {output_path}")
print(f"Laufzeit: {elapsed_time:.2f} Sekunden")

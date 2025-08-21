import pandas as pd
import os
import re

# Basisordner definieren
clean_folder = "03_clean_reviews"
absa_folder = "04.3_absa_files_end2end"
output_folder = "06_final_csv"

os.makedirs(output_folder, exist_ok=True)

# Alle CSV-Files sammeln
clean_files = [f for f in os.listdir(clean_folder) if f.endswith(".csv")]
absa_files = [f for f in os.listdir(absa_folder) if f.endswith(".csv")]

def normalize_name(name: str) -> str:
    """Entfernt Suffixe und normalisiert"""
    name = name.replace(".csv", "").lower()
    name = name.replace("_reviews", "")
    name = name.replace("_absa_aspects", "")
    name = name.replace("_aspects", "")
    name = name.replace("_", "")
    return name


clean_basenames = {normalize_name(f): f for f in clean_files}
absa_basenames = {normalize_name(f): f for f in absa_files}

common_basenames = set(clean_basenames.keys()) & set(absa_basenames.keys())

merged_dfs = []

for base in common_basenames:
    clean_file = clean_basenames[base]
    absa_file = absa_basenames[base]

    clean_path = os.path.join(clean_folder, clean_file)
    absa_path = os.path.join(absa_folder, absa_file)

    df_cleaned = pd.read_csv(clean_path)
    df_aspects = pd.read_csv(absa_path)

    df_merged = pd.merge(
        df_cleaned,
        df_aspects,
        how="left",
        left_on="text_clean",
        right_on="review"
    )
    df_merged.drop(columns=["review"], inplace=True, errors="ignore")
    df_merged["university"] = base

    merged_dfs.append(df_merged)

if merged_dfs:
    df_all = pd.concat(merged_dfs, ignore_index=True)

    cols = [
        "university", "rating_id", "text_full", "text_clean", "rating", "date",
        "aspect", "sentiment"
    ]
    cols_existing = [col for col in cols if col in df_all.columns]
    df_all = df_all[cols_existing + [col for col in df_all.columns if col not in cols_existing]]

    final_path = os.path.join(output_folder, "end2end_final.csv")
    df_all.to_csv(final_path, index=False, encoding="utf-8")
    print("Alle gemergten Dateien wurden zu end2end_final.csv zusammengefügt.")
else:
    print("Keine passenden Dateien gefunden – prüfe, ob die Dateinamen konsistent sind.")

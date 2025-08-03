import pandas as pd
import os

# Basisordner definieren
clean_folder = "03_clean_reviews"
absa_folder = "05_absa_clustered"
output_folder = "06_final_csv"

# Sicherstellen, dass der Output-Ordner existiert
os.makedirs(output_folder, exist_ok=True)

# Alle Cleaned-Files und ABSA-Files sammeln
clean_files = [f for f in os.listdir(clean_folder) if f.endswith(".csv")]
absa_files = [f for f in os.listdir(absa_folder) if f.endswith(".csv")]

# Gemeinsame Basisnamen identifizieren (z. B. "thm" bei "thm_reviews.csv" und "thm_absa_reviews_clustered.csv")
def extract_basename(filename, suffix):
    return filename.replace(suffix, "").replace(".csv", "")

clean_basenames = {extract_basename(f, "_reviews") for f in clean_files}
absa_basenames = {extract_basename(f, "_absa_reviews_clustered") for f in absa_files}

common_basenames = clean_basenames & absa_basenames

# Liste für alle DataFrames
merged_dfs = []

for base in common_basenames:
    clean_path = os.path.join(clean_folder, f"{base}_reviews.csv")
    absa_path = os.path.join(absa_folder, f"{base}_absa_reviews_clustered.csv")

    df_cleaned = pd.read_csv(clean_path)
    df_clustered = pd.read_csv(absa_path)

    df_merged = pd.merge(df_cleaned, df_clustered, how="left", left_on="text_clean", right_on="review")
    df_merged.drop(columns=["review"], inplace=True, errors="ignore")
    df_merged["university"] = base  # Quelle dazuschreiben

    merged_dfs.append(df_merged)

# Alle untereinander hängen
df_all = pd.concat(merged_dfs, ignore_index=True)

# Sortierte Spaltenreihenfolge (optional)
cols = [
    "university", "rating_id", "text_full", "text_clean", "rating", "date",
    "aspect", "sentiment", "cluster", "cluster_name"
]
cols_existing = [col for col in cols if col in df_all.columns]
df_all = df_all[cols_existing + [col for col in df_all.columns if col not in cols_existing]]

# Speichern
final_path = os.path.join(output_folder, "full_absa_dataset.csv")
df_all.to_csv(final_path, index=False, encoding="utf-8")
print("✅ Alle gemergten Dateien wurden zu full_absa_dataset.csv zusammengefügt.")

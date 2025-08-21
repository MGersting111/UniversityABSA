import pandas as pd
from pathlib import Path

reviews_dir = Path("02_reviews")
aspects_dir = Path("05.2_aspects_clustered_llama")
output_path = Path("06_final_csv/llama_final.csv")

def read_all_csvs_in_folder(folder: Path) -> pd.DataFrame:
    all_dfs = []
    for file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="utf-8-sig")
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError(f"Keine CSV-Dateien gefunden in {folder}")
    return pd.concat(all_dfs, ignore_index=True)

df_reviews = read_all_csvs_in_folder(reviews_dir)
df_aspects = read_all_csvs_in_folder(aspects_dir)

# Join-Keys festlegen
if "text_full" in df_reviews.columns and "text_full" in df_aspects.columns:
    left_key, right_key = "text_full", "text_full"
else:
    # Falls "text_full" nicht vorhanden ist, erstes gemeinsames Feld nehmen
    common_cols = [c for c in df_reviews.columns if c in df_aspects.columns]
    if not common_cols:
        raise RuntimeError("Keine gemeinsamen Spalten zum Mergen gefunden.")
    left_key = right_key = common_cols[0]

# LEFT-Join: Alle Reviews bleiben erhalten, ggf. mehrfach pro Aspect
merged = pd.merge(
    df_reviews,
    df_aspects,
    how="left",
    left_on=left_key,
    right_on=right_key,
    suffixes=("_reviews", "_aspects"),
)

merged.to_csv(output_path, index=False, encoding="utf-8")

print(f"{len(df_reviews)} Reviews + {len(df_aspects)} Aspects → {len(merged)} Zeilen")
print(f"Join über {left_key}")
print(f"Gespeichert unter: {output_path}")

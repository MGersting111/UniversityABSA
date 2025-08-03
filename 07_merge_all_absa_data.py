import pandas as pd

# Lade Original-Reviews (text_clean, rating_id, etc.)
df_cleaned = pd.read_csv("03_clean_reviews/frauas_reviews.csv")

# Lade ABSA-Ausgabe (Review, Aspect, Sentiment, Cluster)
df_clustered = pd.read_csv("05_absa_clustered/frauas_absa_reviews_clustered.csv")

# Füge alle Reviews mit ABSA-Zuordnung zusammen (auch leere)
df_final = pd.merge(df_cleaned, df_clustered, how="left", left_on="text_clean", right_on="review")

# Aufräumen: Doppelte Review-Spalte entfernen
df_final.drop(columns=["review"], inplace=True, errors="ignore")

# Sortierte Spaltenreihenfolge
cols = [
    "rating_id", "text_full", "text_clean", "rating", "date",
    "aspect", "sentiment", "cluster", "cluster_name"
]
cols_existing = [col for col in cols if col in df_final.columns]
df_final = df_final[cols_existing + [col for col in df_final.columns if col not in cols_existing]]

# Speichern
df_final.to_csv("06_final_csv/full_absa_dataset.csv", index=False, encoding="utf-8")
print("✅ Alle Reviews (auch ohne Aspekte) gespeichert in full_absa_dataset.csv")

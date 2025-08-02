import pandas as pd
import re

# CSV laden
df = pd.read_csv("02_reviews/tu_darmstadt_reviews.csv")

# Cleaning-Funktion
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zäöüß0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Neue Spalte mit gereinigtem Text
df["text_clean"] = df["text_full"].apply(clean_text)

# Speichern
df.to_csv("03_clean_reviews/tu_darmstadt_reviews.csv", index=False, encoding="utf-8")
print("✅ Gereinigte Reviews gespeichert in cleaned_frankfurtuas_reviews.csv")

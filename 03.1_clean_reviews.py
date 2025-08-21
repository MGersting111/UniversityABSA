import pandas as pd
import re
import unicodedata

df = pd.read_csv("02_reviews/tu_darmstadt_reviews.csv")

def clean_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)  # Unicode-Normalisierung
    text = re.sub(r"http\S+", "", text)         # URLs entfernen
    text = text.replace("\n", " ").replace("\r", " ")  # Zeilenumbrüche
    text = re.sub(r"[•■◆●★►]", "", text)        # Aufzählungszeichen entfernen
    text = re.sub(r"\s+", " ", text)            # Mehrfache Leerzeichen
    return text.strip()

# Neue Spalte mit gereinigtem Text
df["text_clean"] = df["text_full"].apply(clean_text)

# Speichern
df.to_csv("03_clean_reviews/tu_darmstadt_reviews.csv", index=False, encoding="utf-8")
print("Gereinigte Reviews gespeichert in 03_clean_reviews/tu_darmstadt_reviews.csv")

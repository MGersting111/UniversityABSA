from pyabsa import ATEPCCheckpointManager
import pandas as pd
import torch
import json

# CSV laden
df = pd.read_csv("03_clean_reviews/frauas_reviews.csv")

# Lade ABSA-Modell
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint="multilingual",
    auto_device="cuda" if torch.cuda.is_available() else "cpu"
)

# ABSA anwenden
def extract_aspects_from_review(review):
    result = aspect_extractor.extract_aspect(inference_source=[review], print_result=False)
    extracted = []
    if result and result[0]['aspect']:
        for asp, sent in zip(result[0]['aspect'], result[0]['sentiment']):
            extracted.append({
                "aspect": asp,
                "sentiment": sent
            })
    return {
        "review": review,
        "aspects": extracted
    }

output = [extract_aspects_from_review(text) for text in df['text_clean']]

# Flatten für DataFrame
flat_data = []
for entry in output:
    review = entry["review"]
    for asp in entry["aspects"]:
        flat_data.append({
            "review": review,
            "aspect": asp["aspect"],
            "sentiment": asp["sentiment"]
        })

df_flat = pd.DataFrame(flat_data)
df_flat.to_csv("04_absa_files/absa_aspects_frauas.csv", index=False, encoding="utf-8")
print("✅ ABSA-Ergebnisse gespeichert in absa_aspects.csv")

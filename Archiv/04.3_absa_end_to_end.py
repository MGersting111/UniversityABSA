# absa_end_to_end.py
import time
import pandas as pd
import json
from pyabsa import ATEPCCheckpointManager

start_time = time.time()

# CSV laden mit bereinigten Reviews (muss Spalte "text_clean" haben)
df = pd.read_csv("../03_clean_reviews/frauas_reviews.csv")

# Lade das End-to-End ABSA-Modell (Aspekt + Sentiment)
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint="multilingual",    # oder z. B. 'fast_lcf_bert_base'
    auto_device=True              # nutzt automatisch GPU, falls vorhanden
)

# Funktion: extrahiere Aspekte + Sentiment aus einem Review
def extract_aspects_from_review(review):
    result = aspect_extractor.extract_aspect(
        inference_source=[review],
        print_result=False
    )
    extracted = []
    if result and result[0]['aspect']:
        for asp, sent in zip(result[0]['aspect'], result[0]['sentiment']):
            extracted.append({
                "aspect": asp,
                "sentiment": sent
            })
    return extracted

# Flache Struktur: id, Review, Aspect, Sentiment
flat_data = []
review_id = 1

for review in df['text_clean']:
    aspects = extract_aspects_from_review(review)
    if aspects:  # Falls Aspekte gefunden
        for asp in aspects:
            flat_data.append({
                "id": review_id,
                "review": review,
                "aspect": asp["aspect"],
                "sentiment": asp["sentiment"]
            })
    else:  # Falls keine Aspekte gefunden
        flat_data.append({
            "id": review_id,
            "review": review,
            "aspect": None,
            "sentiment": None
        })
    review_id += 1

# In DataFrame umwandeln und speichern
df_out = pd.DataFrame(flat_data)
output_path = "04.3_absa_files_end2end/end2end_fra_uas_absa_aspects.csv"
df_out.to_csv(output_path, index=False, encoding="utf-8")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"✅ End-to-End ABSA-Ergebnisse mit IDs gespeichert in {output_path}")
print(f"Time: {elapsed_time:.2f} Sekunden")

import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Lade ABSA-Modell
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained(
    "yangheng/deberta-v3-base-absa-v1.1"
).to(device)
absa_model.eval()



# Lade deutsches spaCy-Modell
nlp = spacy.load("de_core_news_md")
STOPWORDS = nlp.Defaults.stop_words
BLACKLIST = {"zeit", "sache", "teil", "thema", "jahr", "name", "dinge", "meinung"}

# Lade Reviews
df = pd.read_csv("03_clean_reviews/tu_darmstadt_reviews.csv")

# Aspekte extrahieren (nur Nomen + evtl. ADJ davor)
def extract_aspects(text):
    doc = nlp(text)
    aspects = set()
    for token in doc:
        if token.pos_ == "NOUN" and token.lemma_ not in STOPWORDS and token.lemma_ not in BLACKLIST:
            aspect = token.lemma_
            for child in token.children:
                if child.pos_ == "ADJ" and child.lemma_ not in STOPWORDS:
                    aspect = f"{child.lemma_} {aspect}"
            aspects.add(aspect)
    return list(aspects)

# ABSA-Prediction
def predict_absa(aspect, sentence):
    input_text = f"{aspect} [SEP] {sentence}"
    inputs = absa_tokenizer(input_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).squeeze()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[torch.argmax(probs).item()]

# Hauptpipeline
output = []
for _, row in df.iterrows():
    review = str(row["text_clean"])
    aspects = extract_aspects(review)
    for asp in aspects:
        sentiment = predict_absa(asp, review)
        output.append({
            "review": review,
            "aspect": asp,
            "sentiment": sentiment
        })

# Speichern
df_out = pd.DataFrame(output)
df_out.to_csv("04_absa_files/tu_darmstadt_absa_aspects.csv", index=False, encoding="utf-8")
print("✅ ABSA-Ergebnisse gespeichert in 04_absa_files/tu_darmstadt_absa_aspects.csv")

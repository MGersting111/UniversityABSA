import pandas as pd
import requests
import json
import time
import os

CSV_INPUT = "02_reviews/tu_darmstadt_reviews.csv"
CSV_OUTPUT = "04.2_absa_withllama/tu_darmstadt_review_aspect_sentiment_context.csv"
BATCH_SIZE = 100

# Prompt-Funktion (fokussiert, max. 5 Aspekte, mit Kontextbeschreibung)
def build_prompt(review_text: str) -> str:
    return f"""
Analysiere den folgenden Bewertungstext. Extrahiere maximal 5 relevante Aspekte, die typischerweise in Hochschulbewertungen vorkommen (z. B. Dozenten, Organisation, Studieninhalte, Ausstattung, Campus, Bibliothek, IT, Mensa etc.).

Für jeden Aspekt:
1. Gib das passende Sentiment an (exakt: „positiv“, „neutral“ oder „negativ“ – bitte exakt so schreiben)
2. Gib einen kurzen Begründungssatz oder Beleg aus dem Review, der das Sentiment stützt.
3. Gib den Aspekt als einzelnes Wort oder kurze Phrase (z. B. „Dozenten“, „Organisation“, „Studieninhalte“)

Gib die Antwort **ausschließlich** im folgenden Format als **gültiges JSON-Array** zurück:

[
  {{ "aspect": "Aspekt1", "sentiment": "positiv|neutral|negativ", "context": "Begründung oder Zitat aus Review" }},
  ...
]

Es dürfen **keine Felder fehlen**. Alle Objekte im Array müssen genau diese drei Felder haben: „aspect“, „sentiment“ und „context“.

Text:
\"{review_text.strip()}\"
"""

# Funktion zur Anfrage an lokales Ollama-Modell (API)
def query_ollama(prompt: str, model: str = "mistral") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            return f"ERROR: HTTP {response.status_code}"
    except Exception as e:
        return f"ERROR: {str(e)}"

# JSON aus Modellantwort extrahieren
def extract_aspects_from_output(output: str):
    try:
        start = output.find("[")
        end = output.rfind("]") + 1
        json_text = output[start:end]
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

# Lade bereits gespeicherte Ergebnisse (falls vorhanden)
if os.path.exists(CSV_OUTPUT):
    output_df = pd.read_csv(CSV_OUTPUT)
    existing_ids = set(output_df["review_id"].tolist())
else:
    output_df = pd.DataFrame()
    existing_ids = set()

# Lade Input
full_df = pd.read_csv(CSV_INPUT)
df = full_df[["rating_id", "text_full"]].dropna()
df = df[~df["rating_id"].isin(existing_ids)]
df = df.reset_index(drop=True)

print(f"Starte Analyse von {len(df)} neuen Reviews...")

# Bearbeite in Batches
for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    print(f"Verarbeite Reviews {i} bis {i + len(batch) - 1}...")

    output_rows = []

    for idx, row in batch.iterrows():
        review_id = row["rating_id"]
        review_text = row["text_full"][:500]  # Optional kürzen
        prompt = build_prompt(review_text)

        print(f"Analysiere Review ID {review_id}...")
        response = query_ollama(prompt)
        print(f"Antwort erhalten für Review ID {review_id}.")

        aspects = extract_aspects_from_output(response)

        for item in aspects:
            aspect = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "")).strip().lower()
            context = str(item.get("context", "")).strip()

            if sentiment in ["positive", "positiv"]:
                sentiment = "positiv"
            elif sentiment in ["negative", "negativ"]:
                sentiment = "negativ"
            elif sentiment == "neutral":
                sentiment = "neutral"
            else:
                sentiment = "unbekannt"

            output_rows.append({
                "review_id": review_id,
                "text_full": review_text,
                "aspect": aspect,
                "sentiment": sentiment,
                "context": context
            })

        time.sleep(1)

    # Speichere Batch
    batch_df = pd.DataFrame(output_rows)
    output_df = pd.concat([output_df, batch_df])
    output_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"Batch gespeichert mit {len(batch_df)} Zeilen.")

print("Alle Batches abgeschlossen.")
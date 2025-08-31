# ABSA-Pipeline für Hochschulbewertungen

Dieses Repository enthält eine vollständige Pipeline zur Sammlung, Bereinigung und Analyse von Hochschulbewertungen von **StudyCheck**. Ziel ist die Durchführung einer **Aspect-Based Sentiment Analysis (ABSA)** auf deutschen Reviews, um strukturierte Einblicke in Stärken und Schwächen von Hochschulen zu erhalten.  

Die Pipeline umfasst:
1. Scraping der Review-Daten
2. Vorverarbeitung und Bereinigung
3. Aspect-Based Sentiment Analysis (verschiedene Ansätze)
4. Experimentelle Clustering-Ansätze (nicht final genutzt)
5. Mergen & Finalisierung der Ergebnisse

---

## Installation & Setup

### Voraussetzungen
- Python ≥ 3.9  
- Pakete: `pandas`, `requests`, `beautifulsoup4`, `spacy`, `transformers`, `torch`, `sentence-transformers`, `scikit-learn`, `matplotlib`

Installation der Pakete:
```bash
pip install -r requirements.txt
```

(dein `requirements.txt` sollte aus den Imports aller Skripte generiert werden)

Zusätzlich für spaCy:
```bash
python -m spacy download de_core_news_sm
python -m spacy download de_core_news_md
```

---

## Pipeline-Schritte

### 1. Review-Links scrapen
**Datei:** `01.1_scrape_review_links.py`

- Lädt Review-Links von StudyCheck für eine bestimmte Hochschule (z. B. THM).
- Ergebnisse werden als `.txt` gespeichert.

Output: `01_review_links/review_links_<hochschule>.txt`

---

### 2. Review-Details scrapen
**Datei:** `02_scrape-review_details.py`

- Lädt für jeden Review-Link:
  - Universität, Datum, Text
  - Gesamtbewertung
  - Einzelaspekte (Studieninhalte, Dozenten, Organisation …)
- Speichert alles als CSV.

Output: `02_reviews/<hochschule>_reviews.csv`

---

### 2.1 Studiengang pro Review extrahieren
**Datei:** `02.1_get_subjec_per_review.py`

- Extrahiert den Studiengang aus den Review-Seiten.
- Nutzt mehrere Fallbacks (HTML-Struktur, Breadcrumbs, Seitentitel).
- Speichert Ergebnisse als CSV.

Output: `studiengaenge_<hochschule>.csv`

---

### 3. Reviews bereinigen
**Datei:** `03.1_clean_reviews.py`

- Entfernt:
  - URLs
  - Zeilenumbrüche
  - Aufzählungszeichen
  - Mehrfache Leerzeichen
- Normalisiert Unicode.
- Speichert bereinigte Texte zusätzlich in einer Spalte `text_clean`.

Output: `03_clean_reviews/<hochschule>_reviews.csv`

---

### 4. Aspect-Based Sentiment Analysis (ABSA)

Es wurden mehrere Ansätze getestet:

#### 4.1 Hybrid-Ansatz (spaCy + DeBERTa)
**Datei:** `04.1_absa_hybrid.py`

- Aspekte via Nomen + Adjektive mit spaCy extrahiert.
- Sentiment via DeBERTa-v3 ABSA-Modell klassifiziert.
- Ergebnis: CSV mit Review, Aspekt und Sentiment.

Output: `04.1_absa_files_hybrid/...csv`

---

#### 4.2 ABSA mit LLaMA / Ollama
**Datei:** `04.2_absa_with_llama.py`

- Nutzt lokales LLM (Ollama/Mistral).
- Extrahiert bis zu 5 Aspekte pro Review inkl. Sentiment + Kontextbegründung.
- JSON-Parsing der Modellantworten.
- Ergebnisse werden sukzessive gespeichert.

Output: `04.2_absa_withllama/...csv`

---

#### 4.3 End-to-End PyABSA
**Datei:** `04.3_absa_end_to_end.py`

- Nutzung von PyABSA (multilingual checkpoint).
- Direkte Extraktion von Aspekten + Sentiment aus den bereinigten Reviews.
- Alle Reviews werden in einer großen Datei zusammengeführt.

Output: `04.3.1_absa_files_end2end/end2end_all_aspects.csv`

---

### 5. Experimentelle Cluster-Analysen (nicht final genutzt)

Die folgenden Skripte dienten nur zur Exploration von Aspekt-Clustern, wurden aber nicht für die finale Analyse übernommen:

- `05.1_cluster_aspects.py`
- `05.2_cluster_aspects.py`
- `05.3_cluster_final.py`
- `05.4_cluster_aspects.py`
- `visualisierung_cluster.py`

Ansätze:
- Sentence-BERT Embeddings
- Agglomerative Clustering & KMeans
- Cluster-Benennung via TF-IDF
- Visualisierung mit t-SNE

---

### 6. Mergen der Daten

#### 6.1 End-to-End Merge (PyABSA)
**Datei:** `06.1_merge_all_absa_data.py`

- Merged bereinigte Reviews und ABSA-Ergebnisse.
- Einheitliche CSV `end2end_final.csv` mit allen Hochschulen.

Output: `06_final_csv/end2end_final.csv`

---

#### 6.2 Merge ABSA mit LLaMA-Ergebnissen
**Datei:** `06.2_merge_all_absa_files_llama.py`

- Merged Review-CSV und LLaMA-ABSA-Ergebnisse.
- Robust gegen unterschiedliche Codierungen.
- Ergebnis: kombinierte CSV.

Output: `06_final_csv/llama_final.csv`

---

## Ergebnisse

- Vollständige Reviews mit Metadaten + extrahierten Aspekten + Sentimenten.  
- Unterschiedliche ABSA-Ansätze ermöglichen Vergleichbarkeit.  
- Cluster-Analysen nur explorativ, finale Ergebnisse beruhen auf ABSA ohne Clustering.  

---

## Nutzung

1. Review-Links scrapen:
   ```bash
   python 01.1_scrape_review_links.py
   ```
2. Review-Details extrahieren:
   ```bash
   python 02_scrape-review_details.py
   ```
3. Reviews bereinigen:
   ```bash
   python 03.1_clean_reviews.py
   ```
4. ABSA durchführen (z. B. PyABSA):
   ```bash
   python 04.3_absa_end_to_end.py
   ```
5. Ergebnisse mergen:
   ```bash
   python 06.1_merge_all_absa_data.py
   ```

---

## Hinweis
- Cluster-Analysen (`05.x` + `visualisierung_cluster.py`) waren Experiment und sind nicht in die finale Auswertung eingeflossen.
- Für die eigentliche Analyse wurden die ABSA-Skripte (04.x) + Merge-Skripte (06.x) genutzt.

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import re
import spacy
from collections import defaultdict

CSV_INPUT = "fra_uas_review_aspect_sentiment_context.csv"
CSV_OUTPUT = "clustered_aspects.csv"

# Lade Sprachmodell für Lemmatisierung (Deutsch)
nlp = spacy.load("de_core_news_sm")

# Lemmatisierungsfunktion
def lemmatize(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

# CSV einlesen
df = pd.read_csv(CSV_INPUT)

# Aspekte bereinigen & vereinheitlichen
df["aspect_clean"] = df["aspect"].fillna("").apply(lemmatize)

# Nur eindeutige Aspekte verwenden
unique_aspects = df["aspect_clean"].dropna().unique().tolist()

# Sentence-BERT Embeddings
model = SentenceTransformer("distiluse-base-multilingual-cased")
embeddings = model.encode(unique_aspects, show_progress_bar=True)

# Clustering mit AgglomerativeClustering
n_clusters = 50  # kannst du anpassen
affinity = "euclidean"
linkage = "ward"

cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
cluster_labels = cluster_model.fit_predict(embeddings)

# Cluster-Labels den Aspekten zuweisen
aspect_clusters = pd.DataFrame({
    "aspect_clean": unique_aspects,
    "cluster": cluster_labels
})

# Cluster benennen mit Top-N Begriffen
cluster_names = {}
vectorizer = CountVectorizer(max_features=1000, stop_words="english")

for cluster_id in sorted(aspect_clusters["cluster"].unique()):
    cluster_terms = aspect_clusters[aspect_clusters["cluster"] == cluster_id]["aspect_clean"]
    X = vectorizer.fit_transform(cluster_terms)
    terms = vectorizer.get_feature_names_out()
    scores = np.array(X.sum(axis=0)).flatten()
    top_n = 3
    top_terms = [terms[i] for i in scores.argsort()[::-1][:top_n]]
    cluster_names[cluster_id] = ", ".join(top_terms)

# Cluster-Namen zuweisen
aspect_clusters["cluster_name"] = aspect_clusters["cluster"].map(cluster_names)

# Mit ursprünglichem DataFrame zusammenführen
merged_df = df.merge(aspect_clusters, on="aspect_clean", how="left")

# Speichern
merged_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
print(f"Cluster gespeichert in {CSV_OUTPUT}")

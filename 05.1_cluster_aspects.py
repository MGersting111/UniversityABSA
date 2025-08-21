import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy

CSV_INPUT = "04.1_absa_files_hybrid/tu_darmstadt_absa_aspects.csv"
CSV_OUTPUT = "OLD__absa_clustered/tu_darmstadt_absa_reviews_clustered.csv"

df = pd.read_csv(CSV_INPUT, encoding="utf-8")
aspects = df["aspect"].dropna().unique().tolist()

nlp = spacy.load("de_core_news_md")
german_stopwords = list(nlp.Defaults.stop_words)

#Embeddings erzeugen
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(aspects, show_progress_bar=True)

#Beste Clusteranzahl finden
best_k = 2
best_score = -1
for k in range(2, min(21, len(aspects))):
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    if score > best_score:
        best_score = score
        best_k = k

#Finales Clustering
final_model = AgglomerativeClustering(n_clusters=best_k)
labels = final_model.fit_predict(embeddings)

#Cluster zuordnen
clustered_df = pd.DataFrame({
    "aspect": aspects,
    "cluster": labels
})

#Automatische Benennung
cluster_names = {}
aspects_per_cluster = clustered_df.groupby("cluster")["aspect"].apply(list)

for cluster_id, asp_list in aspects_per_cluster.items():
    vec = CountVectorizer(stop_words=german_stopwords, ngram_range=(1, 2))
    X = vec.fit_transform(asp_list)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    top_term = terms[np.argmax(scores)] if len(terms) > 0 else f"Cluster {cluster_id}"
    cluster_names[cluster_id] = top_term

clustered_df["cluster_name"] = clustered_df["cluster"].map(cluster_names)

#Merge mit Originaldaten
df_merged = pd.merge(df, clustered_df, on="aspect", how="left")

#Speichern
df_merged.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
print(f"Cluster-Ergebnisse gespeichert in: {CSV_OUTPUT}")

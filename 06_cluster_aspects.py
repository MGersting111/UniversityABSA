from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sentence_transformers import SentenceTransformer



# Lade ABSA-Daten
df = pd.read_csv("absa_aspects_thm.csv")
unique_aspects = df['aspect'].dropna().unique().tolist()

# Embedding-Modell laden
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(unique_aspects)

# Clustering
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Cluster DataFrame
clustered = pd.DataFrame({
    'aspect': unique_aspects,
    'cluster': labels
})

# === Automatische Cluster-Benennung ===
cluster_names = {}
aspects_per_cluster = clustered.groupby("cluster")["aspect"].apply(list)

for cluster_id, aspects in aspects_per_cluster.items():
    if len(aspects) == 1:
        cluster_names[cluster_id] = aspects[0]
        continue

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(aspects)
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top_term = terms[scores.argmax()]
    cluster_names[cluster_id] = top_term

# Cluster-Namen zuordnen
clustered["cluster_name"] = clustered["cluster"].map(cluster_names)

# Merge mit ABSA-Daten
df_merged = pd.merge(df, clustered, on="aspect", how="left")

# Speichern
df_merged.to_csv("absa_reviews_with_clusters_thm.csv", index=False)
print("✅ Reviews mit Clustern und automatisch benannten Gruppen gespeichert in absa_reviews_with_clusters.csv")

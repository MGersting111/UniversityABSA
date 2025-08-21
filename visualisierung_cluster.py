import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

INPUT_CSV = "04.1_absa_files_hybrid/tu_darmstadt_absa_aspects.csv"

df = pd.read_csv(INPUT_CSV, encoding="utf-8")
aspects = (
    pd.Series(df["aspect"].dropna().astype(str).str.strip())
    .loc[lambda s: s.ne("")].unique().tolist()
)

if len(aspects) < 5:
    raise ValueError("Zu wenige Aspekte für eine sinnvolle Visualisierung.")

print(f"Aspekte: {len(aspects)} einzigartig")

#Embeddings
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
X = np.asarray(model.encode(aspects, show_progress_bar=True))

#  k automatisch grob bestimmen (Silhouette auf kleinem Raster)
def pick_k_silhouette(X, k_min=3, k_max=20, random_state=42, n_init=10):
    k_max = min(k_max, max(k_min, X.shape[0] - 1))
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

k = pick_k_silhouette(X, k_min=3, k_max=20)
print(f"Gewählte Clusteranzahl (Silhouette): k={k}")

# KMeans-Clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

#2D-Projektion für Visualisierung (t-SNE)
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
X_2d = tsne.fit_transform(X)

# Plotten
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=35, alpha=0.85)
plt.title("Aspekt-Embeddings (t-SNE) mit KMeans-Clustern", pad=12)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# einige Punkte beschriften (z. B. pro Cluster die 5 häufigsten Aspekte im Datensatz)
counts = df["aspect"].value_counts()
top_per_cluster = 5
for cl in np.unique(labels):
    idxs = [i for i, lab in enumerate(labels) if lab == cl]
    # sortiere Clusterpunkte nach globaler Häufigkeit, nimm Top-N
    idxs_sorted = sorted(idxs, key=lambda i: counts.get(aspects[i], 0), reverse=True)[:top_per_cluster]
    for i in idxs_sorted:
        plt.annotate(aspects[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.9)

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

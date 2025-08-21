from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os

INPUT_FILE = "04.1_absa_files_hybrid/frauas_absa_aspects.csv"
OUTPUT_FILE = "05.1_aspects_clustered_hybrid/fra_uas_clustered.csv"


df = pd.read_csv(INPUT_FILE)
unique_aspects = pd.Series(df["aspect"].dropna().unique()).tolist()
if len(unique_aspects) < 2:
    raise ValueError("Zu wenige unterschiedliche Aspekte für Clustering (mind. 2 benötigt).")

#Embeddings
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
X = np.asarray(model.encode(unique_aspects, show_progress_bar=False))

#Deutsche Stopwörter via spaCy (mit Fallback)
def load_german_stopwords():
    try:
        import spacy
        try:
            nlp = spacy.load("de_core_news_md")
        except OSError:
            # Falls nur small installiert ist
            nlp = spacy.load("de_core_news_sm")
        return list(nlp.Defaults.stop_words)
    except Exception:
        # kleiner Fallback-Satz
        return list({
            "aber","als","am","an","auch","auf","aus","bei","bin","bis","bist","da","dadurch","daher","darum",
            "das","dass","dein","deine","dem","den","der","des","deshalb","die","dies","dieser","dieses","doch",
            "dort","du","durch","ein","eine","einem","einen","einer","eines","er","es","euer","eure","für",
            "hatte","hatten","hattest","hattet","hier","hinter","ich","ihr","ihre","im","in","ist","ja","jede",
            "jedem","jeden","jeder","jedes","jener","jenes","jetzt","kann","kein","keine","keinem","keinen",
            "keiner","keines","können","könnte","machen","man","manche","manchem","manchen","mancher","manches",
            "mein","meine","mit","muss","musste","nach","nicht","nichts","noch","nun","nur","ob","oder","ohne",
            "sehr","sein","seine","sich","sie","sind","so","solche","solchem","solchen","solcher","solches",
            "soll","sollte","sondern","sonst","um","und","uns","unser","unsere","unter","vom","von","vor",
            "wann","warum","was","weiter","weitere","wenn","wer","werde","werden","wie","wieder","will","wir",
            "wird","wirst","wo","wollen","wollte","würde","würden","zu","zum","zur","über"
        })

GERMAN_STOPWORDS = load_german_stopwords()

# k bestimmen: kombinierter Score (Silhouette + CH - DB), k>=3
def choose_optimal_k(
    X,
    k_min=3,
    k_max=50,
    random_state=42,
    n_init=10,
    weights=(0.5, 0.35, 0.15)  # Silhouette, CH, (invertiertes) DB
):
    n_samples = X.shape[0]
    if n_samples < 3:
        return 2 if n_samples == 2 else 1
    k_max = max(k_min, min(k_max, n_samples - 1))

    rows = []
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = km.fit_predict(X)
            if len(set(labels)) < k:
                continue
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            rows.append((k, sil, ch, db))
        except Exception:
            continue

    if not rows:
        return 2 if n_samples >= 2 else 1

    dfm = pd.DataFrame(rows, columns=["k", "sil", "ch", "db"])

    # Standardisieren
    def z(x):
        mu, sd = x.mean(), x.std(ddof=0)
        return (x - mu) / (sd if sd > 0 else 1.0)

    score = (
        weights[0] * z(dfm["sil"]) +
        weights[1] * z(dfm["ch"]) +
        weights[2] * z(-dfm["db"])  # DB: kleiner besser
    )

    # De-priorisiere sehr kleine k bei vielen Aspekten
    if n_samples >= 30:
        score = score.where(dfm["k"] != 2, score - 0.75)
    if n_samples >= 60:
        score = score.where(dfm["k"] != 3, score - 0.25)

    dfm["combo"] = score
    best_score = dfm["combo"].max()
    best_k = int(dfm.loc[dfm["combo"] >= best_score - 0.05, "k"].min())  # bei Gleichstand kleineres k
    print("Top-k (kombinierter Score):")
    print(dfm.sort_values("combo", ascending=False).head(5))
    print(f"Gewählte Clusteranzahl k={best_k}")
    return best_k

n_clusters = choose_optimal_k(X, k_min=3, k_max=50)

#Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

clustered = pd.DataFrame({"aspect": unique_aspects, "cluster": labels})

# Automatische Cluster-Benennung (TF-IDF + deutsche Stopwörter)
cluster_names = {}
aspects_per_cluster = clustered.groupby("cluster")["aspect"].apply(list)

for cluster_id, aspects in aspects_per_cluster.items():
    if len(aspects) == 1:
        cluster_names[cluster_id] = aspects[0]
        continue

    vectorizer = TfidfVectorizer(stop_words=GERMAN_STOPWORDS, ngram_range=(1, 2))
    X_txt = vectorizer.fit_transform(aspects)
    scores = X_txt.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_term = terms[scores.argmax()] if scores.max() > 0 else aspects[0]
    cluster_names[cluster_id] = top_term

clustered["cluster_name"] = clustered["cluster"].map(cluster_names)

# === Merge & Speichern ===
df_merged = pd.merge(df, clustered, on="aspect", how="left")


df_merged.to_csv(OUTPUT_FILE, index=False)
print(f"Reviews mit {n_clusters} Clustern und automatisch benannten Gruppen gespeichert in {OUTPUT_FILE}")

import pandas as pd
from sentence_transformers import SentenceTransformer
import plotly.express as px
import umap
import spacy


CSV_INPUT = "04.1_absa_files_hybrid/tu_darmstadt_absa_aspects.csv"
CSV_OUTPUT = "OLD__absa_clustered/tu_darmstadt_absa_reviews_clustered.csv"

df = pd.read_csv(CSV_INPUT, encoding="utf-8")
aspects = df["aspect"].dropna().unique().tolist()

#Lade deutsche Stopwords
nlp = spacy.load("de_core_news_md")
german_stopwords = list(nlp.Defaults.stop_words)

#Embeddings erzeugen
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(aspects, show_progress_bar=True)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
X2 = reducer.fit_transform(embeddings)
labels = None
color = labels if labels is not None else None
fig = px.scatter(
    x=X2[:,0], y=X2[:,1],
    color=color,
    hover_name=aspects,      # zeigt den Aspekt-Text beim Hover
    title="Aspekte-Embeddings in 2D (UMAP)"
)
fig.update_traces(marker=dict(size=6, opacity=0.9))
fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2")
fig.show()
import streamlit as st
import pandas as pd

# CSV laden
df = pd.read_csv("06_final_csv/full_absa_dataset.csv", parse_dates=["date"], dayfirst=True)

# --- Sidebar-Filter ---
st.sidebar.header("🔍 Filter")

# Universität
uni_options = df["university"].unique().tolist()
selected_unis = st.sidebar.multiselect("Universität", uni_options, default=uni_options)

# Zeitraum
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Zeitraum", [min_date, max_date])

# Cluster
cluster_options = df["cluster_name"].dropna().unique().tolist()
selected_clusters = st.sidebar.multiselect("Cluster", cluster_options, default=cluster_options)

# Sentiment
sentiment_options = df["sentiment"].dropna().unique().tolist()
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)

# --- Daten filtern ---
filtered_df = df[
    (df["university"].isin(selected_unis)) &
    (df["cluster_name"].isin(selected_clusters)) &
    (df["sentiment"].isin(selected_sentiments)) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# --- Dashboard Titel ---
st.title("🎓 Uni-Review Dashboard (ABSA-basiert)")

# --- Durchschnittliches Rating ---
st.header("📈 Durchschnittliches Rating pro Universität")
avg_rating = filtered_df.groupby("university")["rating"].mean()
st.bar_chart(avg_rating)

# --- Bewertung über Zeit ---
st.header("📆 Entwicklung der Bewertungen über Zeit")
time_rating = filtered_df.groupby(["date", "university"])["rating"].mean().unstack()
st.line_chart(time_rating)

# --- Aspekte ---
st.header("📚 Aspekte im Vergleich")
aspect_cols = [
    "Studieninhalte", "Dozenten", "Lehrveranstaltungen", "Ausstattung",
    "Organisation", "Literaturzugang", "Digitales Studieren"
]
aspect_avg = filtered_df.groupby("university")[aspect_cols].mean()
st.bar_chart(aspect_avg)

# --- Sentiment-Verteilung ---
st.header("💬 Sentiment-Verteilung")
sentiment_count = filtered_df.groupby(["sentiment", "university"]).size().unstack(fill_value=0)
st.bar_chart(sentiment_count)

# --- Cluster-Verteilung ---
st.header("🧠 Cluster-Verteilung")
cluster_count = filtered_df.groupby(["cluster_name", "university"]).size().unstack(fill_value=0)
st.bar_chart(cluster_count)

# --- Optional: Rohdaten anzeigen ---
with st.expander("🔎 Rohdaten anzeigen"):
    st.dataframe(filtered_df)

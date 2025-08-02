import streamlit as st
import pandas as pd
import altair as alt

@st.cache_data
def load_data():
    return pd.read_csv("full_absa_dataset.csv")

df = load_data()

st.set_page_config(layout="wide")
st.title("🎓 Uni Review Dashboard")

col_filter, col_main, col_kpis = st.columns([1, 3, 1])

with col_filter:
    st.sidebar.header("Filter")

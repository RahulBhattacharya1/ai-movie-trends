import os
import math
import pandas as pd
import numpy as np
import streamlit as st

# Optional imports for clustering/search
try:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_csv(path, empty_cols=None):
    if not os.path.exists(path):
        if empty_cols:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=empty_cols or [])

@st.cache_data
def load_data():
    movies = load_csv(os.path.join(DATA_DIR, "movies_clean.csv"))
    agg_year = load_csv(os.path.join(DATA_DIR, "agg_ratings_by_year.csv"))
    pca_2d = load_csv(os.path.join(DATA_DIR, "pca_2d.csv"))
    return movies, agg_year, pca_2d

@st.cache_resource
def load_vectorizer_kmeans():
    vec_path = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
    km_path = os.path.join(DATA_DIR, "kmeans.pkl")
    if joblib and os.path.exists(vec_path) and os.path.exists(km_path):
        vectorizer = joblib.load(vec_path)
        kmeans = joblib.load(km_path)
        return vectorizer, kmeans
    return None, None

def page_header():
    st.title("Movie Trends Explorer")
    st.caption("Interactive trends, clusters, and semantic search from your dataset.")

def sidebar_filters(movies):
    # Year filter
    years = movies["year"].dropna().astype(int) if "year" in movies.columns else pd.Series([], dtype=int)
    if len(years) > 0:
        yr_min, yr_max = int(years.min()), int(years.max())
        year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
    else:
        year_range = None

    # Rating filter
    if "rating" in movies.columns and movies["rating"].notna().any():
        rmin = float(movies["rating"].min())
        rmax = float(movies["rating"].max())
        rating_min, rating_max = st.sidebar.slider(
            "Rating range",
            min_value=math.floor(rmin*10)/10,
            max_value=math.ceil(rmax*10)/10,
            value=(math.floor(rmin*10)/10, math.ceil(rmax*10)/10)
        )
    else:
        rating_min, rating_max = None, None

    return year_range, (rating_min, rating_max)

def apply_filters(movies, year_range, rating_range):
    df = movies.copy()
    if year_range and "year" in df.columns:
        df = df[(df["year"]>=year_range[0]) & (df["year"]<=year_range[1])]
    if rating_range and "rating" in df.columns:
        df = df[(df["rating"].isna()) | ((df["rating"]>=rating_range[0]) & (df["rating"]<=rating_range[1]))]
    return df

def draw_overview_tab(movies_filt, agg_year):
    st.subheader("Overview")
    st.write("Basic dataset summary based on current filters.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Movies", len(movies_filt))
    with c2:
        if "rating" in movies_filt.columns and movies_filt["rating"].notna().any():
            st.metric("Avg Rating", f"{movies_filt['rating'].mean():.2f}")
        else:
            st.metric("Avg Rating", "N/A")
    with c3:
        if "votes" in movies_filt.columns and movies_filt["votes"].notna().any():
            st.metric("Median Votes", int(movies_filt["votes"].median()))
        else:
            st.metric("Median Votes", "N/A")

    st.markdown("---")
    if not agg_year.empty and "year" in agg_year.columns:
        st.subheader("Ratings by Year")
        import altair as alt
        chart_data = agg_year.dropna().copy()
        line = alt.Chart(chart_data).mark_line().encode(
            x="year:Q",
            y=alt.Y("avg_rating:Q", title="Average Rating")
        )
        bars = alt.Chart(chart_data).mark_bar(opacity=0.25).encode(
            x="year:Q",
            y=alt.Y("count:Q", title="Count"),
        )
        st.altair_chart(
            alt.layer(bars, line).resolve_scale(y="independent").properties(height=320),
            use_container_width=True
        )
    else:
        st.info("No year information available.")

def draw_clusters_tab(movies, pca_2d):
    st.subheader("Clusters (KMeans on TF-IDF of overview)")
    if pca_2d.empty or "cluster" not in pca_2d.columns:
        st.info("Clustering not available. Provide overview text and regenerate artifacts in Colab.")
        return
    import altair as alt
    pca_2d["cluster"] = pca_2d["cluster"].astype(int)
    chart = alt.Chart(pca_2d).mark_circle(size=40).encode(
        x="x:Q", y="y:Q",
        color="cluster:N",
        tooltip=["title","cluster"]
    ).properties(height=520)
    st.altair_chart(chart, use_container_width=True)

def draw_search_tab(movies, vectorizer, kmeans):
    st.subheader("Semantic Search")
    if vectorizer is None:
        st.info("Vectorizer not available. Generate tfidf_vectorizer.pkl in Colab.")
        return
    query = st.text_input("Describe a movie you want to find (e.g., space adventure with strong female lead)")
    n = st.slider("Results", 5, 50, 10)
    if query:
        texts = movies["overview"].fillna("").astype(str).tolist()
        X = vectorizer.transform(texts)
        q = vectorizer.transform([query])
        sims = cosine_similarity(q, X).ravel()
        idx = np.argsort(-sims)[:n]
        cols = [c for c in ["title","year","rating","overview"] if c in movies.columns]
        res = movies.iloc[idx][cols].copy()
        res["similarity"] = sims[idx]
        st.dataframe(res)

def main():
    page_header()
    movies, agg_year, pca_2d = load_data()
    vectorizer, kmeans = load_vectorizer_kmeans()

    with st.sidebar:
        st.header("Filters")
        yr_rng, rating_rng = sidebar_filters(movies)

    movies_filt = apply_filters(movies, yr_rng, rating_rng)

    tabs = st.tabs(["Overview", "Clusters", "Search"])
    with tabs[0]:
        draw_overview_tab(movies_filt, agg_year)
    with tabs[1]:
        draw_clusters_tab(movies, pca_2d)
    with tabs[2]:
        draw_search_tab(movies, vectorizer, kmeans)

if __name__ == "__main__":
    main()

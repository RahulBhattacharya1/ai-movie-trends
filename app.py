import os
import math
import pandas as pd
import numpy as np
import streamlit as st

# Optional imports for clustering/search
try:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
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
    genre_exploded = load_csv(os.path.join(DATA_DIR, "genre_exploded.csv"))
    pca_2d = load_csv(os.path.join(DATA_DIR, "pca_2d.csv"))
    return movies, agg_year, genre_exploded, pca_2d

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
    st.caption("Interactive trends, genre breakdowns, clusters, and semantic search from your dataset.")

def sidebar_filters(movies, genre_exploded):
    years = movies["year"].dropna().astype(int) if "year" in movies.columns else pd.Series([], dtype=int)
    if len(years) > 0:
        yr_min, yr_max = int(years.min()), int(years.max())
        year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
    else:
        year_range = None

    all_genres = []
    if "genres" in genre_exploded.columns:
        all_genres = sorted([g for g in genre_exploded["genres"].dropna().unique() if str(g).strip()!=""])
    selected_genres = st.sidebar.multiselect("Genres", options=all_genres, default=[])

    if "rating" in movies.columns and movies["rating"].notna().any():
        rmin = float(movies["rating"].min())
        rmax = float(movies["rating"].max())
        rating_min, rating_max = st.sidebar.slider("Rating range", min_value=math.floor(rmin*10)/10,
                                                   max_value=math.ceil(rmax*10)/10,
                                                   value=(math.floor(rmin*10)/10, math.ceil(rmax*10)/10))
    else:
        rating_min, rating_max = None, None

    return year_range, selected_genres, (rating_min, rating_max)

def apply_filters(movies, genre_exploded, year_range, selected_genres, rating_range):
    df = movies.copy()
    if year_range and "year" in df.columns:
        df = df[(df["year"]>=year_range[0]) & (df["year"]<=year_range[1])]
    if rating_range and "rating" in df.columns:
        df = df[(df["rating"].isna()) | ((df["rating"]>=rating_range[0]) & (df["rating"]<=rating_range[1]))]
    if selected_genres:
        ge = genre_exploded.copy()
        ge = ge[ge["genres"].isin(selected_genres)]
        keep_titles = set(ge["title"].dropna().astype(str).tolist())
        df = df[df["title"].astype(str).isin(keep_titles)]
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
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=320), use_container_width=True)
    else:
        st.info("No year information available.")

def draw_genre_tab(genre_exploded, movies_filt):
    st.subheader("Genres")
    if genre_exploded.empty:
        st.info("No genres available.")
        return
    ge = genre_exploded.copy()
    if not movies_filt.empty:
        keep_titles = set(movies_filt["title"].astype(str))
        ge = ge[ge["title"].astype(str).isin(keep_titles)]
    top = ge.groupby("genres", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).head(25)

    import altair as alt
    chart = alt.Chart(top).mark_bar().encode(
        x="count:Q",
        y=alt.Y("genres:N", sort="-x", title="Genre"),
        tooltip=["genres","count"]
    ).properties(height=480)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Sample Titles")
    sel_genre = st.selectbox("Pick a genre to preview titles", options=top["genres"].tolist())
    if sel_genre:
        sample = ge[ge["genres"]==sel_genre].drop_duplicates(subset=["title"]).head(50)
        cols = [c for c in ["title","year","rating","overview"] if c in sample.columns]
        st.dataframe(sample[cols])

def ensure_text(movies):
    if "overview" in movies.columns and movies["overview"].fillna("").str.strip().any():
        return True
    return False

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
        # Vectorize corpus on the fly if needed; better is to reuse training vectorizer
        # Here we reuse vectorizer and transform the movie overviews again for cosine similarity
        texts = movies["overview"].fillna("").astype(str).tolist()
        X = vectorizer.transform(texts)
        q = vectorizer.transform([query])
        sims = cosine_similarity(q, X).ravel()
        idx = np.argsort(-sims)[:n]
        res = movies.iloc[idx][["title","year","rating","overview"] if "rating" in movies.columns else ["title","year","overview"]].copy()
        res["similarity"] = sims[idx]
        st.dataframe(res)

def main():
    page_header()
    movies, agg_year, genre_exploded, pca_2d = load_data()
    vectorizer, kmeans = load_vectorizer_kmeans()

    with st.sidebar:
        st.header("Filters")
        yr_rng, sel_genres, rating_rng = sidebar_filters(movies, genre_exploded)

    movies_filt = apply_filters(movies, genre_exploded, yr_rng, sel_genres, rating_rng)

    tabs = st.tabs(["Overview", "Genres", "Clusters", "Search"])
    with tabs[0]:
        draw_overview_tab(movies_filt, agg_year)
    with tabs[1]:
        draw_genre_tab(genre_exploded, movies_filt)
    with tabs[2]:
        draw_clusters_tab(movies, pca_2d)
    with tabs[3]:
        draw_search_tab(movies, vectorizer, kmeans)

if __name__ == "__main__":
    main()

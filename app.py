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


# ------------------------------
# Utilities: loading and caching
# ------------------------------
def load_csv(path, empty_cols=None):
    """Load CSV if present; otherwise return empty DataFrame with optional schema."""
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
    """Load all app data files."""
    movies = load_csv(os.path.join(DATA_DIR, "movies_clean.csv"))
    agg_year = load_csv(os.path.join(DATA_DIR, "agg_ratings_by_year.csv"))
    genre_exploded = load_csv(os.path.join(DATA_DIR, "genre_exploded.csv"))
    pca_2d = load_csv(os.path.join(DATA_DIR, "pca_2d.csv"))
    return movies, agg_year, genre_exploded, pca_2d


@st.cache_resource
def load_vectorizer_kmeans():
    """Load optional TF-IDF vectorizer and KMeans model for semantic search/clusters."""
    vec_path = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
    km_path = os.path.join(DATA_DIR, "kmeans.pkl")
    if joblib and os.path.exists(vec_path) and os.path.exists(km_path):
        vectorizer = joblib.load(vec_path)
        kmeans = joblib.load(km_path)
        return vectorizer, kmeans
    return None, None


# ------------------------------
# Genre handling (robust)
# ------------------------------
def _split_genres_flexible(s):
    """Split a genre cell into a list, accepting many formats and separators."""
    if pd.isna(s):
        return []
    t = str(s).strip()
    if not t:
        return []
    # Try to parse list-like strings: ["Action","Drama"] or ['Action', 'Drama']
    if t.startswith("[") and t.endswith("]"):
        try:
            val = eval(t)
            return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
    # Normalize multiple possible separators into commas
    for sep in ["|", ";", "/", "\\", ">", "<"]:
        t = t.replace(sep, ",")
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return parts


def build_all_genres(movies_df, genre_exploded_df):
    """
    Return a sorted list of genres for the sidebar.
    Strategy:
      1) Prefer 'genre_exploded.csv' -> 'genres' column.
      2) If missing/empty/Unknown-only, derive from movies via common columns.
      3) If still empty, return ['All'] as a fallback so dropdown never empty.
    """
    # 1) Use genre_exploded if valid
    if isinstance(genre_exploded_df, pd.DataFrame) and not genre_exploded_df.empty:
        if "genres" in genre_exploded_df.columns:
            vals = genre_exploded_df["genres"].dropna().astype(str).str.strip()
            # Drop placeholders like Unknown/empty
            vals = vals.replace("Unknown", np.nan)
            vals = vals[vals != ""]
            uniq = sorted(vals.dropna().unique().tolist())
            if len(uniq) > 0:
                return uniq

    # 2) Derive genres from movies if possible
    if isinstance(movies_df, pd.DataFrame) and not movies_df.empty:
        candidates = [c for c in ["genres", "genres_raw", "genre", "Genre"] if c in movies_df.columns]
        all_genres = []
        for col in candidates:
            col_vals = movies_df[col].fillna("").astype(str).tolist()
            for s in col_vals:
                all_genres.extend(_split_genres_flexible(s))
        uniq = sorted({g for g in all_genres if g and g.lower() != "unknown"})
        if len(uniq) > 0:
            return uniq

    # 3) Fallback so the UI is never empty
    return ["All"]


def row_matches_selected_genres(row, selected_genres, possible_cols):
    """Check whether a row matches any selected genres using flexible splitting."""
    # If user selected 'All', we treat it as no genre filtering.
    if "All" in selected_genres:
        return True
    for c in possible_cols:
        vals = _split_genres_flexible(row.get(c, ""))
        if any(g in vals for g in selected_genres):
            return True
    return False


# ------------------------------
# Sidebar + Filtering
# ------------------------------
def page_header():
    st.title("Movie Trends Explorer")
    st.caption("Interactive trends, genre breakdowns, clusters, and semantic search from your dataset.")


def sidebar_filters(movies, genre_exploded):
    # Year range
    years = movies["year"].dropna().astype(int) if "year" in movies.columns else pd.Series([], dtype=int)
    if len(years) > 0:
        yr_min, yr_max = int(years.min()), int(years.max())
        year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
    else:
        year_range = None

    # Genres (robust)
    all_genres = build_all_genres(movies, genre_exploded)
    # If we only have 'All', set it as default; otherwise show an empty default selection.
    default_genres = ["All"] if all_genres == ["All"] else []
    selected_genres = st.sidebar.multiselect("Genres", options=all_genres, default=default_genres)

    # Rating range
    if "rating" in movies.columns and movies["rating"].notna().any():
        rmin = float(movies["rating"].min())
        rmax = float(movies["rating"].max())
        rating_min, rating_max = st.sidebar.slider(
            "Rating range",
            min_value=math.floor(rmin * 10) / 10,
            max_value=math.ceil(rmax * 10) / 10,
            value=(math.floor(rmin * 10) / 10, math.ceil(rmax * 10) / 10),
        )
    else:
        rating_min, rating_max = None, None

    return year_range, selected_genres, (rating_min, rating_max)


def apply_filters(movies, genre_exploded, year_range, selected_genres, rating_range):
    df = movies.copy()

    # Year filter
    if year_range and "year" in df.columns:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Rating filter
    if rating_range and "rating" in df.columns:
        df = df[(df["rating"].isna()) | ((df["rating"] >= rating_range[0]) & (df["rating"] <= rating_range[1]))]

    # Genre filter
    if selected_genres and "All" not in selected_genres:
        # Prefer exploded if valid; else derive per-row
        if (
            isinstance(genre_exploded, pd.DataFrame)
            and not genre_exploded.empty
            and "genres" in genre_exploded.columns
        ):
            ge = genre_exploded.copy()
            ge["genres"] = ge["genres"].astype(str).str.strip()
            ge = ge[~ge["genres"].isin(["", "Unknown"])]
            ge = ge[ge["genres"].isin(selected_genres)]
            keep_titles = set(ge["title"].dropna().astype(str).tolist())
            df = df[df["title"].astype(str).isin(keep_titles)]
        else:
            possible_cols = [c for c in ["genres", "genres_raw", "genre", "Genre"] if c in df.columns]
            if possible_cols:
                df = df[df.apply(lambda r: row_matches_selected_genres(r, selected_genres, possible_cols), axis=1)]
            else:
                # No columns to determine genres; keep df as-is rather than hiding everything.
                pass

    return df


# ------------------------------
# Tabs and visualizations
# ------------------------------
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
            try:
                st.metric("Median Votes", int(movies_filt["votes"].median()))
            except Exception:
                st.metric("Median Votes", "N/A")
        else:
            st.metric("Median Votes", "N/A")

    st.markdown("---")
    if isinstance(agg_year, pd.DataFrame) and not agg_year.empty and "year" in agg_year.columns:
        st.subheader("Ratings by Year")
        try:
            import altair as alt
            chart_data = agg_year.dropna().copy()
            line = alt.Chart(chart_data).mark_line().encode(
                x="year:Q",
                y=alt.Y("avg_rating:Q", title="Average Rating"),
            )
            bars = alt.Chart(chart_data).mark_bar(opacity=0.25).encode(
                x="year:Q",
                y=alt.Y("count:Q", title="Count"),
            )
            st.altair_chart(
                alt.layer(bars, line).resolve_scale(y="independent").properties(height=320),
                use_container_width=True,
            )
        except Exception:
            st.info("Altair could not render the chart. Check that 'altair' is installed and data is valid.")
    else:
        st.info("No year information available.")


def draw_genre_tab(genre_exploded, movies_filt):
    st.subheader("Genres")

    # Build a safe working frame
    ge = pd.DataFrame()
    if isinstance(genre_exploded, pd.DataFrame) and not genre_exploded.empty:
        ge = genre_exploded.copy()
        if "genres" in ge.columns:
            ge["genres"] = ge["genres"].astype(str).str.strip()
            ge = ge[ge["genres"].notna()]
        else:
            ge["genres"] = []
    else:
        ge["genres"] = []

    # Filter to only movies in current selection (if any)
    if isinstance(movies_filt, pd.DataFrame) and not movies_filt.empty and "title" in ge.columns:
        keep_titles = set(movies_filt["title"].astype(str))
        ge = ge[ge["title"].astype(str).isin(keep_titles)]

    # Remove placeholders/empties
    ge = ge[~ge["genres"].isin(["", "Unknown"])]

    if ge.empty:
        st.info("No genre information available to display.")
        return

    # Top genres
    top = ge.groupby("genres", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).head(25)

    try:
        import altair as alt
        chart = alt.Chart(top).mark_bar().encode(
            x="count:Q",
            y=alt.Y("genres:N", sort="-x", title="Genre"),
            tooltip=["genres", "count"],
        ).properties(height=480)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(top)

    st.markdown("### Sample Titles")
    sel_genre = st.selectbox("Pick a genre to preview titles", options=top["genres"].tolist())
    if sel_genre and "title" in ge.columns:
        sample = ge[ge["genres"] == sel_genre].drop_duplicates(subset=["title"]).head(50)
        cols = [c for c in ["title", "year", "rating", "overview"] if c in sample.columns]
        if cols:
            st.dataframe(sample[cols])
        else:
            st.dataframe(sample)


def draw_clusters_tab(movies, pca_2d):
    st.subheader("Clusters (KMeans on TF-IDF of overview)")
    if not isinstance(pca_2d, pd.DataFrame) or pca_2d.empty or "cluster" not in pca_2d.columns:
        st.info("Clustering not available. Provide overview text and regenerate artifacts in Colab.")
        return
    try:
        import altair as alt
        pca_2d = pca_2d.copy()
        pca_2d["cluster"] = pca_2d["cluster"].astype(int)
        chart = alt.Chart(pca_2d).mark_circle(size=40).encode(
            x="x:Q",
            y="y:Q",
            color="cluster:N",
            tooltip=["title", "cluster"],
        ).properties(height=520)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(pca_2d.head(100))


def draw_search_tab(movies, vectorizer, kmeans):
    st.subheader("Semantic Search")
    if vectorizer is None or not isinstance(movies, pd.DataFrame) or movies.empty:
        st.info("Semantic search unavailable. Ensure tfidf_vectorizer.pkl exists and movies data is loaded.")
        return

    query = st.text_input("Describe a movie you want to find (e.g., space adventure with strong female lead)")
    n = st.slider("Results", 5, 50, 10)

    if query:
        texts = movies["overview"].fillna("").astype(str).tolist() if "overview" in movies.columns else []
        if not any(len(t.strip()) > 0 for t in texts):
            st.info("Movies do not contain overview text to search over.")
            return
        X = vectorizer.transform(texts)
        q = vectorizer.transform([query])
        sims = cosine_similarity(q, X).ravel()
        idx = np.argsort(-sims)[:n]
        cols = [c for c in ["title", "year", "rating", "overview"] if c in movies.columns]
        res = movies.iloc[idx][cols].copy()
        res["similarity"] = sims[idx]
        st.dataframe(res)


# ------------------------------
# Main
# ------------------------------
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

# Movie Trends Explorer

Interactive dashboard to explore movie trends (ratings by year, genre breakdowns), plus optional clustering and semantic search on overviews.

## Setup

1. Upload your raw CSV to `data/movies.csv`.
2. Open the Colab notebook steps (see README top). Run cells to generate:
   - `data/movies_clean.csv`
   - `data/agg_ratings_by_year.csv`
   - `data/genre_exploded.csv`
   - `data/tfidf_vectorizer.pkl` (optional)
   - `data/kmeans.pkl` (optional)
   - `data/pca_2d.csv` (optional)
3. Commit and push to GitHub.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

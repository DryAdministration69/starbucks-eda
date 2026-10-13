# Starbucks EDA — Streamlit App

An interactive Exploratory Data Analysis (EDA) dashboard for the Starbucks beverage nutrition dataset.

## Features
- Dataset overview, summary statistics, and missing-value checks
- Filters for beverage category and preparation/size
- Nutrition comparisons and category-level averages
- Correlation heatmap for numeric nutrition metrics
- Filtered CSV download
- An educational decision-tree classifier for beverage category

## Files
- `app.py` — Streamlit application
- `starbucks.csv` — dataset
- `starbucks-eda.ipynb` — original Kaggle notebook
- `requirements.txt` — Python dependencies

## Run locally
Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push all project files to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** and select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

Keep `app.py`, `requirements.txt`, and `starbucks.csv` in the repository root unless you update the paths accordingly.

## Dataset notes
The app trims extra whitespace from column names. Missing-value counts are based on pandas' recognized blank/NA values. The classification model is for educational exploration only; its results can be affected by the small dataset and by using beverage names/preparation labels as predictors.

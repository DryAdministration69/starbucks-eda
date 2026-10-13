import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Starbucks EDA", page_icon="☕", layout="wide")

st.title("☕ Starbucks Nutrition Explorer")
st.write(
    "Explore nutritional information for Starbucks beverages, compare drinks, "
    "visualize patterns, and view a simple beverage-category classification model."
)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("starbucks.csv")
    # Remove accidental leading/trailing spaces in column names.
    data.columns = data.columns.astype(str).str.strip()
    return data

uploaded = st.sidebar.file_uploader("Optional: upload a Starbucks CSV", type=["csv"])
try:
    df = load_data(uploaded)
except FileNotFoundError:
    st.error("Could not find starbucks.csv. Add it to the same GitHub repository as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

st.sidebar.header("Explore filters")
if "Beverage_category" in df.columns:
    categories = sorted(df["Beverage_category"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect(
        "Beverage category", categories, default=categories
    )
else:
    selected_categories = []

filtered = df.copy()
if "Beverage_category" in filtered.columns and selected_categories:
    filtered = filtered[filtered["Beverage_category"].astype(str).isin(selected_categories)]
elif "Beverage_category" in filtered.columns and not selected_categories:
    filtered = filtered.iloc[0:0]

if "Beverage_prep" in filtered.columns:
    preparations = sorted(filtered["Beverage_prep"].dropna().astype(str).unique())
    selected_preps = st.sidebar.multiselect("Preparation / size", preparations, default=preparations)
    filtered = filtered[filtered["Beverage_prep"].astype(str).isin(selected_preps)]

numeric_candidates = [
    "Calories", "Total Fat (g)", "Trans Fat (g)", "Saturated Fat (g)",
    "Sodium (mg)", "Total Carbohydrates (g)", "Cholesterol (mg)",
    "Dietary Fibre (g)", "Sugars (g)", "Protein (g)", "Caffeine (mg)"
]
for col in numeric_candidates:
    if col in df.columns:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset rows", f"{len(df):,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Filtered drinks", f"{len(filtered):,}")
m4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

tab_overview, tab_charts, tab_data, tab_model = st.tabs(
    ["Overview", "Visual analysis", "Data table", "Decision tree"]
)

with tab_overview:
    st.subheader("Dataset overview")
    st.write("This dataset contains nutrition details and preparation information for Starbucks beverages.")
    left, right = st.columns(2)
    with left:
        st.markdown("**First rows**")
        st.dataframe(df.head(10), use_container_width=True)
    with right:
        st.markdown("**Missing values by column**")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No blank/NA values were detected by pandas.")
        else:
            miss_df = missing.rename_axis("Column").reset_index(name="Missing values")
            st.dataframe(miss_df, use_container_width=True)
            st.plotly_chart(px.bar(miss_df, x="Column", y="Missing values", title="Missing values"), use_container_width=True)
    st.markdown("**Numeric summary**")
    st.dataframe(df.describe(include="number").T, use_container_width=True)

with tab_charts:
    st.subheader("Compare nutrition values")
    available_numeric = [c for c in numeric_candidates if c in filtered.columns]
    if not available_numeric or filtered.empty:
        st.info("Choose at least one category and preparation in the sidebar to see charts.")
    else:
        chosen_metric = st.selectbox("Nutrition metric", available_numeric, index=0)
        if "Beverage" in filtered.columns:
            chart_df = filtered.copy()
            chart_df["Drink label"] = chart_df["Beverage"].astype(str)
            if "Beverage_prep" in chart_df.columns:
                chart_df["Drink label"] += " — " + chart_df["Beverage_prep"].astype(str)
            chart_df = chart_df.sort_values(chosen_metric, ascending=False)
            st.plotly_chart(
                px.bar(chart_df, x="Drink label", y=chosen_metric, color="Beverage_category" if "Beverage_category" in chart_df.columns else None,
                       title=f"{chosen_metric} by drink", hover_data=[c for c in ["Beverage_category", "Beverage_prep"] if c in chart_df.columns]),
                use_container_width=True
            )
            st.caption("For readability, use the table filters or select fewer categories/preparations when many drinks are shown.")
        if "Beverage_category" in filtered.columns:
            summary = filtered.groupby("Beverage_category", dropna=False)[chosen_metric].mean().reset_index()
            summary = summary.sort_values(chosen_metric, ascending=False)
            st.plotly_chart(px.bar(summary, x="Beverage_category", y=chosen_metric,
                                   title=f"Average {chosen_metric} by beverage category"),
                            use_container_width=True)
        if len(available_numeric) > 1:
            corr_cols = [c for c in available_numeric if filtered[c].notna().any()]
            if len(corr_cols) > 1:
                st.plotly_chart(px.imshow(filtered[corr_cols].corr(numeric_only=True), text_auto=".2f",
                                          aspect="auto", title="Correlation between nutrition metrics"),
                                use_container_width=True)

with tab_data:
    st.subheader("Filtered data")
    st.dataframe(filtered, use_container_width=True, height=450)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="starbucks_filtered.csv",
        mime="text/csv",
    )

with tab_model:
    st.subheader("Decision tree: predict beverage category")
    st.warning(
        "This is a small educational dataset. The model uses beverage names and preparation labels "
        "as features, so its accuracy should not be interpreted as real-world performance on unseen drink types."
    )
    target = "Beverage_category"
    if target not in df.columns:
        st.error(f"Target column '{target}' is missing.")
    else:
        model_df = df.dropna(subset=[target]).copy()
        y = model_df[target].astype(str)
        X = model_df.drop(columns=[target])
        # Remove columns that are entirely empty.
        X = X.dropna(axis=1, how="all")
        categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical = X.select_dtypes(include="number").columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical),
            ],
            remainder="drop"
        )
        test_size = st.slider("Test data proportion", min_value=0.15, max_value=0.40, value=0.20, step=0.05)
        max_depth = st.slider("Maximum tree depth", min_value=2, max_value=8, value=4)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(max_depth=max_depth, random_state=42))
            ])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            st.metric("Test accuracy", f"{score:.1%}")
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T
            st.markdown("**Classification report**")
            st.dataframe(report_df.round(3), use_container_width=True)
            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, predictions, labels=labels)
            st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto",
                                      labels=dict(x="Predicted category", y="Actual category", color="Count"),
                                      title="Confusion matrix"), use_container_width=True)
        except ValueError as exc:
            st.error(f"The model could not be trained with the current split: {exc}")

st.caption("Project: Starbucks EDA | Built with Streamlit, pandas, Plotly, and scikit-learn.")

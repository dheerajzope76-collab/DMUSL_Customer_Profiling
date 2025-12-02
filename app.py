
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# Paths and global objects
# ---------------------------

MODEL_PATH = Path("random_forest_smote_model.pkl")
DATA_PATH = Path("Customer_Profiling (1).csv")

rf_model = None
kmeans_model = None
scaler = None

CLUSTER_LABELS = {
    0: "Engaged & Established",
    1: "Graduated & Balanced Spenders",
    2: "High-Income, PhD, High-Response Spenders",
}

CLUSTER_STRATEGIES = {
    "Engaged & Established": (
        "Middle-aged, family-heavy, moderately affluent customers with Master's degrees.\\n\\n"
        "- Focus on family-oriented bundles, value-for-money offers, and retention.\\n"
        "- Use a mix of email + SMS + occasional personalized offers.\\n"
        "- Emphasize stability, convenience, and trust."
    ),
    "Graduated & Balanced Spenders": (
        "Mainstream, mass-market segment with graduation-level education and balanced spending.\\n\\n"
        "- Good for broad campaigns and cross-sell (sweets, gold, general lifestyle products).\\n"
        "- Use multi-channel campaigns (email, app notifications, social media).\\n"
        "- Offer loyalty points, combo deals, and seasonal promotions."
    ),
    "High-Income, PhD, High-Response Spenders": (
        "Small but premium segment: high income, PhD, older, high wine expenditure, and most responsive.\\n\\n"
        "- Prioritize for premium, high-margin campaigns (wine, gourmet, curated experiences).\\n"
        "- Use highly personalized communication (email, RM, curated offers).\\n"
        "- Offer exclusivity: VIP programs, early access, limited editions."
    ),
}


# ---------------------------
# Utility: load models safely
# ---------------------------

def load_random_forest_model():
    global rf_model
    if rf_model is None and MODEL_PATH.exists():
        rf_model = joblib.load(MODEL_PATH)
    return rf_model


def fit_kmeans_from_data():
    """
    Fit KMeans on engineered features using the original dataset,
    so we can assign new customers to the same 3 clusters.
    \"""
    global kmeans_model, scaler

    if not DATA_PATH.exists():
        return None, None

    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature engineering to mirror your notebook
    # Age
    if "Year_Birth" in df.columns:
        df["Age"] = 2024 - df["Year_Birth"]
    else:
        df["Age"] = np.nan

    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["Total_Expenditure"] = df[mnt_cols].sum(axis=1)

    # Approximation: divide by 24 months for average monthly spend
    df["Average_Monthly_Spend"] = df["Total_Expenditure"] / 24.0

    # Engagement score: total purchase counts
    df["Engagement_Score"] = (
        df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    # Dependents
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]

    # Campaign_Response (use Response column if present)
    if "Response" in df.columns:
        df["Campaign_Response"] = df["Response"]
    else:
        df["Campaign_Response"] = 0

    # Education dummies
    edu = df["Education"].str.lower()
    df["Education_Basic"] = edu.str.contains("basic").astype(int)
    df["Education_Graduation"] = (
        edu.str.contains("graduat").astype(int)
        & (~edu.str.contains("post"))
    )
    df["Education_Master"] = (
        edu.str.contains("master") | edu.str.contains("postgrad")
    ).astype(int)
    df["Education_PhD"] = edu.str.contains("phd").astype(int)

    feature_cols = [
        "Income",
        "Age",
        "Recency",
        "Total_Expenditure",
        "Average_Monthly_Spend",
        "Engagement_Score",
        "Dependents",
        "Campaign_Response",
        "Education_Basic",
        "Education_Graduation",
        "Education_Master",
        "Education_PhD",
    ]

    df_features = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)

    return kmeans_model, scaler


def ensure_kmeans():
    global kmeans_model, scaler
    if kmeans_model is None or scaler is None:
        kmeans_model, scaler = fit_kmeans_from_data()
    return kmeans_model, scaler


# ---------------------------
# Feature engineering for NEW input
# ---------------------------

def engineer_features_from_input(user_input: dict) -> pd.DataFrame:
    \"""
    user_input: dict with raw inputs from the UI
    Returns a single-row DataFrame with engineered features as in clustering.
    \"""
    # Create base DataFrame
    df = pd.DataFrame([user_input])

    # Total expenditure
    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["Total_Expenditure"] = df[mnt_cols].sum(axis=1)

    df["Average_Monthly_Spend"] = df["Total_Expenditure"] / 24.0

    df["Engagement_Score"] = (
        df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    df["Dependents"] = df["Kidhome"] + df["Teenhome"]

    # Campaign_Response is unknown for new customers; keep as 0 for clustering
    df["Campaign_Response"] = 0

    # Education dummies
    edu = df["Education"].str.lower()
    df["Education_Basic"] = (edu == "basic").astype(int)
    df["Education_Graduation"] = (edu == "graduation").astype(int)
    df["Education_Master"] = (edu == "master").astype(int)
    df["Education_PhD"] = (edu == "phd").astype(int)

    feature_cols = [
        "Income",
        "Age",
        "Recency",
        "Total_Expenditure",
        "Average_Monthly_Spend",
        "Engagement_Score",
        "Dependents",
        "Campaign_Response",
        "Education_Basic",
        "Education_Graduation",
        "Education_Master",
        "Education_PhD",
    ]

    return df[feature_cols]


def assign_cluster(features_df: pd.DataFrame):
    \"""
    Use KMeans to assign cluster to the new customer based on engineered features.
    \"""
    kmeans, sc = ensure_kmeans()
    if kmeans is None or sc is None:
        return None, None, None

    X_scaled = sc.transform(features_df)
    cluster_id = int(kmeans.predict(X_scaled)[0])

    label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
    strategy = CLUSTER_STRATEGIES.get(label, "No specific strategy defined.")
    return cluster_id, label, strategy


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="Campaign Response Prediction & Customer Segmentation",
        layout="centered",
    )

    st.title("🎯 Campaign Response Prediction & Customer Segmentation")

    st.markdown(
        \"""
This app uses a **Random Forest (with SMOTE)** model to predict whether a customer is
**likely to respond** to a marketing campaign, and then assigns them to one of three
customer **segments (clusters)** with recommended marketing strategies.
\"""
    )

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.markdown(
        \"""
- Model: **Random Forest Classifier (with SMOTE)**
- Segmentation: **K-Means (k=3)** on behavioral + demographic features  
- Goal: Identify **who will respond**, and **how to market** to them.
\"
    )

    st.header("🧾 Enter Customer Details")

    # Layout: 2 columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            max_value=2000000.0,
            value=50000.0,
            step=1000.0,
        )
        st.caption("Typical range: 0 – 200000 (currency units)")

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=55,
            step=1,
        )
        st.caption("Typical range: 25 – 75 years")

        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0,
            max_value=365,
            value=50,
            step=1,
        )
        st.caption("Typical range: 0 – 100 days")

        kidhome = st.number_input(
            "Number of kids at home",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
        )
        st.caption("Typical range: 0 – 3")

        teenhome = st.number_input(
            "Number of teenagers at home",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
        )
        st.caption("Typical range: 0 – 3")

        education = st.selectbox(
            "Education level",
            options=["Basic", "Graduation", "Master", "PhD"],
            index=1,
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=["Single", "Together", "Married", "Divorced", "Widow"],
            index=2,
        )

    with col2:
        mnt_wines = st.number_input(
            "Amount spent on Wines (last 2 years)",
            min_value=0.0,
            max_value=20000.0,
            value=500.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 2000")

        mnt_fruits = st.number_input(
            "Amount spent on Fruits (last 2 years)",
            min_value=0.0,
            max_value=5000.0,
            value=50.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 500")

        mnt_meat = st.number_input(
            "Amount spent on Meat Products (last 2 years)",
            min_value=0.0,
            max_value=20000.0,
            value=200.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 2000")

        mnt_fish = st.number_input(
            "Amount spent on Fish Products (last 2 years)",
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 1000")

        mnt_sweet = st.number_input(
            "Amount spent on Sweet Products (last 2 years)",
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 1000")

        mnt_gold = st.number_input(
            "Amount spent on Gold Products (last 2 years)",
            min_value=0.0,
            max_value=50000.0,
            value=200.0,
            step=10.0,
        )
        st.caption("Typical range: 0 – 5000")

        num_web_purch = st.number_input(
            "Number of Web Purchases",
            min_value=0,
            max_value=100,
            value=3,
            step=1,
        )
        st.caption("Typical range: 0 – 20")

        num_catalog_purch = st.number_input(
            "Number of Catalog Purchases",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
        )
        st.caption("Typical range: 0 – 20")

        num_store_purch = st.number_input(
            "Number of Store Purchases",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )
        st.caption("Typical range: 0 – 20")

        num_web_visits = st.number_input(
            "Number of Web Visits per Month",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )
        st.caption("Typical range: 0 – 10")

    st.markdown("---")

    if st.button("Predict Response & Segment Customer"):
        # Prepare input row
        user_row = {
            "Income": income,
            "Age": age,
            "Recency": recency,
            "Kidhome": kidhome,
            "Teenhome": teenhome,
            "Education": education,
            "Marital_Status": marital_status,
            "MntWines": mnt_wines,
            "MntFruits": mnt_fruits,
            "MntMeatProducts": mnt_meat,
            "MntFishProducts": mnt_fish,
            "MntSweetProducts": mnt_sweet,
            "MntGoldProds": mnt_gold,
            "NumWebPurchases": num_web_purch,
            "NumCatalogPurchases": num_catalog_purch,
            "NumStorePurchases": num_store_purch,
            "NumWebVisitsMonth": num_web_visits,
        }

        input_df_raw = pd.DataFrame([user_row])

        # Load RF model
        model = load_random_forest_model()
        if model is None:
            st.error(
                "❌ Could not load 'random_forest_smote_model.pkl'. "
                "Please ensure it is in the same folder as app.py."
            )
            return

        # NOTE: This assumes your pickle is a pipeline that can handle the columns in input_df_raw.
        # If not, adapt 'input_df_raw' or your pipeline accordingly.
        try:
            pred_class = int(model.predict(input_df_raw)[0])
            # If predict_proba available, show probability
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df_raw)[0][1]
            else:
                prob = None
        except Exception as e:
            st.error(
                "Error when calling model.predict(). "
                "Make sure your model pipeline expects these columns.\\n\\n"
                f"Details: {e}"
            )
            return

        # Cluster assignment based on engineered features
        features_df = engineer_features_from_input(user_row)
        cluster_id, cluster_label, strategy_text = assign_cluster(features_df)

        st.header("📊 Prediction Results")

        if pred_class == 1:
            st.success("✅ This customer is **LIKELY to respond** to the campaign.")
        else:
            st.warning("⚠️ This customer is **NOT very likely to respond** to the campaign.")

        if prob is not None:
            st.markdown(
                f"**Estimated probability of positive response:** `{prob:.2%}`"
            )

        if cluster_label is not None:
            st.subheader(f"🧩 Segment Assignment: {cluster_label}")
            st.markdown(
                f"**Cluster ID:** `{cluster_id}`\\n\\n"
                f"**Segment Description & Strategy:**\\n\\n{strategy_text}"
            )
        else:
            st.info(
                "Cluster model could not be loaded. "
                "Ensure 'Customer_Profiling (1).csv' is present to enable segmentation."
            )


if __name__ == "__main__":
    main()

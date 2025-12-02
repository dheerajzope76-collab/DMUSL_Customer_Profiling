import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    "Engaged & Established": """
**Profile**
- Middle-aged customers with families and moderate income  
- Reasonable but not very high spending, especially on daily-use categories  
- Often Master’s educated, value stability and family security  

**Marketing Strategy**
- Pitch **family packs, bundle discounts, and subscription offers** for everyday products  
- Use **SMS + email** with clear savings messaging (“save X per month for your family”)  
- Promote **festive / school season offers** that help with household budgeting  
- Encourage app usage with **small but frequent rewards** (cashback, loyalty points)
""",
    "Graduated & Balanced Spenders": """
**Profile**
- Large “mass” segment with graduation-level education  
- Balanced spending across categories like sweets, gold, meat, etc.  
- Similar age to Cluster 0 but a bit more discretionary spend  

**Marketing Strategy**
- Run **broad lifestyle campaigns**: weekend offers, combo deals, cross-category bundles  
- Use **multi-channel communication** – app notifications, email, social media  
- Push **loyalty tiers** and “spend more, earn more” style programs  
- Offer **personalized recommendations** across multiple categories to increase basket size
""",
    "High-Income, PhD, High-Response Spenders": """
**Profile**
- Highest income, PhD-educated, older and very mature financially  
- Strong preference for **premium wines** and higher total spend  
- Highest historical campaign response rate  

**Marketing Strategy**
- Treat as **elite / VIP customers** – focus on **high-margin, premium products**  
- Use **highly personalized communication**: curated email journeys, RM / concierge support  
- Invite to **exclusive events** (wine tasting, curated experiences, early access launches)  
- Offer **limited edition products, priority delivery, and invite-only loyalty tiers**
""",
}


def load_rf():
    global rf_model
    if rf_model is None and MODEL_PATH.exists():
        rf_model = joblib.load(MODEL_PATH)
    return rf_model


def fit_kmeans():
    global kmeans_model, scaler
    if not DATA_PATH.exists():
        return None, None

    df = pd.read_csv(DATA_PATH)
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2024 - df["Year_Birth"]

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
    df["Campaign_Response"] = df["Response"]

    edu = df["Education"].str.lower()
    df["Education_Basic"] = (edu == "basic").astype(int)
    df["Education_Graduation"] = (edu == "graduation").astype(int)
    df["Education_Master"] = (edu == "master").astype(int)
    df["Education_PhD"] = (edu == "phd").astype(int)

    cols = [
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])

    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)
    return kmeans_model, scaler


def ensure_kmeans():
    global kmeans_model, scaler
    if kmeans_model is None or scaler is None:
        kmeans_model, scaler = fit_kmeans()
    return kmeans_model, scaler


def engineer_features(user_dict):
    df = pd.DataFrame([user_dict])

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
    df["Campaign_Response"] = 0

    edu = df["Education"].str.lower()
    df["Education_Basic"] = (edu == "basic").astype(int)
    df["Education_Graduation"] = (edu == "graduation").astype(int)
    df["Education_Master"] = (edu == "master").astype(int)
    df["Education_PhD"] = (edu == "phd").astype(int)

    cols = [
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

    return df[cols]


def build_features_for_model(user_dict, model):
    base = engineer_features(user_dict)
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return base
    for col in expected:
        if col not in base.columns:
            base[col] = 0.0
    base = base[list(expected)]
    return base


def assign_cluster(feat_df):
    kmeans, sc = ensure_kmeans()
    if kmeans is None or sc is None:
        return None, None, None
    Xs = sc.transform(feat_df)
    cid = int(kmeans.predict(Xs)[0])
    label = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
    strat = CLUSTER_STRATEGIES.get(label, "")
    return cid, label, strat


def add_custom_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            padding-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 15px;
            color: #bbbbbb;
            padding-bottom: 1rem;
        }
        .card {
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-top: 1rem;
            background-color: #111827;
            border: 1px solid #1f2933;
        }
        .card-success {
            border-left: 4px solid #10b981;
        }
        .card-warning {
            border-left: 4px solid #f59e0b;
        }
        .card-segment {
            border-left: 4px solid #3b82f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Campaign Response Prediction & Segmentation",
        layout="centered",
    )

    add_custom_css()

    st.markdown('<div class="main-title">🎯 Campaign Response Prediction & Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Score a customer for campaign response and instantly see which segment they belong to and how to market to them.</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("How to use")
    st.sidebar.write(
        "- Fill in customer profile and spend details\n"
        "- Click **Predict** to see response likelihood\n"
        "- Review **segment profile** and **marketing strategy**"
    )

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Annual Income", 0.0, 2000000.0, 50000.0)
        age = st.number_input("Age (years)", 18, 100, 55)
        recency = st.number_input("Recency (days since last purchase)", 0, 365, 50)
        kidhome = st.number_input("Kids at Home", 0, 5, 0)
        teenhome = st.number_input("Teens at Home", 0, 5, 0)
        edu = st.selectbox("Education", ["Basic", "Graduation", "Master", "PhD"])
        marital = st.selectbox(
            "Marital Status", ["Single", "Together", "Married", "Divorced", "Widow"]
        )

    with col2:
        mw = st.number_input("MntWines (last 2 years)", 0.0, 20000.0, 500.0)
        mf = st.number_input("MntFruits (last 2 years)", 0.0, 5000.0, 50.0)
        mm = st.number_input("MntMeatProducts (last 2 years)", 0.0, 20000.0, 200.0)
        mfsh = st.number_input("MntFishProducts (last 2 years)", 0.0, 10000.0, 50.0)
        msw = st.number_input("MntSweetProducts (last 2 years)", 0.0, 10000.0, 50.0)
        mg = st.number_input("MntGoldProds (last 2 years)", 0.0, 50000.0, 200.0)
        nwp = st.number_input("NumWebPurchases", 0, 100, 3)
        ncp = st.number_input("NumCatalogPurchases", 0, 100, 2)
        nsp = st.number_input("NumStorePurchases", 0, 100, 5)
        nwv = st.number_input("NumWebVisitsMonth", 0, 100, 5)

    st.markdown("---")

    if st.button("🚀 Predict Response & Segment"):
        user = {
            "Income": income,
            "Age": age,
            "Recency": recency,
            "Kidhome": kidhome,
            "Teenhome": teenhome,
            "Education": edu,
            "Marital_Status": marital,
            "MntWines": mw,
            "MntFruits": mf,
            "MntMeatProducts": mm,
            "MntFishProducts": mfsh,
            "MntSweetProducts": msw,
            "MntGoldProds": mg,
            "NumWebPurchases": nwp,
            "NumCatalogPurchases": ncp,
            "NumStorePurchases": nsp,
            "NumWebVisitsMonth": nwv,
        }

        model = load_rf()
        if model is None:
            st.error("Model file 'random_forest_smote_model.pkl' not found.")
            return

        X_model = build_features_for_model(user, model)

        try:
            pred = int(model.predict(X_model)[0])
            prob = (
                model.predict_proba(X_model)[0][1]
                if hasattr(model, "predict_proba")
                else None
            )
        except Exception as e:
            st.error(f"Error when calling model.predict: {e}")
            return

        feat_for_cluster = engineer_features(user)
        cid, label, strat = assign_cluster(feat_for_cluster)

        if pred == 1:
            card_class = "card card-success"
            msg = "✅ This customer is **LIKELY** to respond to the campaign."
        else:
            card_class = "card card-warning"
            msg = "⚠️ This customer is **NOT very likely** to respond to the campaign."

        st.markdown(f'<div class="{card_class}"><h3>Prediction</h3><p>{msg}</p></div>', unsafe_allow_html=True)

        if prob is not None:
            st.markdown(
                f'<div class="{card_class}"><p><b>Estimated probability of positive response:</b> {prob:.2%}</p></div>',
                unsafe_allow_html=True,
            )

        if label is not None:
            st.markdown(
                f'<div class="card card-segment"><h3>Segment</h3><p><b>{label}</b> (Cluster {cid})</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card card-segment">{CLUSTER_STRATEGIES.get(label, "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No cluster assignment available (KMeans not fitted).")


if __name__ == "__main__":
    main()

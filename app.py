
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
    "Engaged & Established": (
        "Middle-aged, family-heavy, moderately affluent customers.\n"
        "- Family bundles and value-driven campaigns.\n"
        "- Balanced communication: SMS/Email."
    ),
    "Graduated & Balanced Spenders": (
        "Mass-market, balanced spenders.\n"
        "- Broad campaigns, lifestyle and combo offers.\n"
        "- Multi-channel communication."
    ),
    "High-Income, PhD, High-Response Spenders": (
        "Premium, high-income, highly responsive customers.\n"
        "- Personalized premium offers.\n"
        "- VIP programs and curated experiences."
    ),
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

    mnt_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Expenditure"] = df[mnt_cols].sum(axis=1)
    df["Average_Monthly_Spend"] = df["Total_Expenditure"] / 24.0
    df["Engagement_Score"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]
    df["Campaign_Response"] = df["Response"]

    edu = df["Education"].str.lower()
    df["Education_Basic"] = (edu == "basic").astype(int)
    df["Education_Graduation"] = (edu == "graduation").astype(int)
    df["Education_Master"] = (edu == "master").astype(int)
    df["Education_PhD"] = (edu == "phd").astype(int)

    cols = [
        "Income","Age","Recency","Total_Expenditure","Average_Monthly_Spend",
        "Engagement_Score","Dependents","Campaign_Response",
        "Education_Basic","Education_Graduation","Education_Master","Education_PhD"
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

    mnt_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Expenditure"] = df[mnt_cols].sum(axis=1)
    df["Average_Monthly_Spend"] = df["Total_Expenditure"] / 24.0
    df["Engagement_Score"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]
    df["Campaign_Response"] = 0

    edu = df["Education"].str.lower()
    df["Education_Basic"] = (edu == "basic").astype(int)
    df["Education_Graduation"] = (edu == "graduation").astype(int)
    df["Education_Master"] = (edu == "master").astype(int)
    df["Education_PhD"] = (edu == "phd").astype(int)

    cols = [
        "Income","Age","Recency","Total_Expenditure","Average_Monthly_Spend",
        "Engagement_Score","Dependents","Campaign_Response",
        "Education_Basic","Education_Graduation","Education_Master","Education_PhD"
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

def main():
    st.title("Campaign Response Prediction & Segmentation")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Income", 0.0, 2000000.0, 50000.0)
        age = st.number_input("Age", 18, 100, 55)
        recency = st.number_input("Recency (days since last purchase)", 0, 365, 50)
        kidhome = st.number_input("Kids at Home", 0, 5, 0)
        teenhome = st.number_input("Teens at Home", 0, 5, 0)
        edu = st.selectbox("Education", ["Basic","Graduation","Master","PhD"])
        marital = st.selectbox("Marital Status", ["Single","Together","Married","Divorced","Widow"])

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

    if st.button("Predict"):
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
            prob = model.predict_proba(X_model)[0][1] if hasattr(model, "predict_proba") else None
        except Exception as e:
            st.error(f"Error when calling model.predict: {e}")
            return

        feat_for_cluster = engineer_features(user)
        cid, label, strat = assign_cluster(feat_for_cluster)

        st.subheader("Prediction")
        if pred == 1:
            st.write("Likely to respond to the campaign.")
        else:
            st.write("Not very likely to respond to the campaign.")
        if prob is not None:
            st.write(f"Estimated probability: {prob:.2%}")

        if label is not None:
            st.subheader("Segment")
            st.write(f"{label} (Cluster {cid})")
            if strat:
                st.write(strat)
        else:
            st.write("No cluster assignment available (KMeans not fitted).")

if __name__ == "__main__":
    main()

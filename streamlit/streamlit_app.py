# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import requests
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import scipy.sparse

st.set_page_config(page_title="UPI Fraud Detection Dashboard", layout="wide")

# ---------------------------
# Config / Helpers
# ---------------------------
DEFAULT_API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000/predict")
MODEL_FILE = "fraud_model_xgb.pkl"
PREPROC_FILE = "preprocessor.pkl"

@st.cache_resource
def load_model_and_preprocessor():
    model = None
    preprocessor = None
    if os.path.exists(MODEL_FILE) and os.path.exists(PREPROC_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(PREPROC_FILE, "rb") as f:
            preprocessor = pickle.load(f)
    return model, preprocessor

def preprocess_df(df, preprocessor):
    # ensure expected columns order
    required_cols = [
        "transaction_id","user_id","amount","time_hour",
        "location_match","num_txn_last_24h","avg_amount_last_7d",
        "is_blacklisted_merchant","transaction_type",
        "merchant_category","device_type"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[required_cols].copy()
    X = preprocessor.transform(df)
    # if sparse matrix -> toarray
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return X

def predict_local(df, model, preprocessor):
    X = preprocess_df(df, preprocessor)
    probs = model.predict_proba(X)[:,1]
    preds = (probs > 0.5).astype(int)
    return probs, preds

def predict_api_row(row, api_url):
    payload = {
        "transaction_id": int(row["transaction_id"]),
        "user_id": int(row["user_id"]),
        "amount": float(row["amount"]),
        "time_hour": int(row["time_hour"]),
        "location_match": int(row["location_match"]),
        "num_txn_last_24h": int(row["num_txn_last_24h"]),
        "avg_amount_last_7d": float(row["avg_amount_last_7d"]),
        "is_blacklisted_merchant": int(row["is_blacklisted_merchant"]),
        "transaction_type": str(row["transaction_type"]),
        "merchant_category": str(row["merchant_category"]),
        "device_type": str(row["device_type"])
    }
    try:
        r = requests.post(api_url, json=payload, timeout=5)
        r.raise_for_status()
        res = r.json()
        return float(res.get("fraud_probability", 0.0)), int(res.get("is_fraud", 0))
    except Exception as e:
        st.error(f"API request failed: {e}")
        return 0.0, 0

# ---------------------------
# Load model/preprocessor if available
# ---------------------------
model, preprocessor = load_model_and_preprocessor()

# ---------------------------
# Sidebar: mode & upload
# ---------------------------
st.sidebar.title("Controls")
mode = st.sidebar.selectbox("Prediction Mode", ["Local (pickle)", "API"])
if mode == "API":
    api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)
else:
    api_url = None

st.sidebar.markdown("**Upload transactions CSV** (must contain required columns).")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Quick example download
if st.sidebar.button("Download example CSV"):
    example = pd.DataFrame({
        "transaction_id":[1001,1002],
        "user_id":[2001,2002],
        "amount":[25000,120.5],
        "time_hour":[2,14],
        "location_match":[0,1],
        "num_txn_last_24h":[18,1],
        "avg_amount_last_7d":[1500,100.0],
        "is_blacklisted_merchant":[0,0],
        "transaction_type":["P2P","P2M"],
        "merchant_category":["Shopping","Food"],
        "device_type":["Android","iOS"]
    })
    csv = example.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download example CSV", csv, "example_upi.csv", "text/csv")

# ---------------------------
# Main layout
# ---------------------------
st.title("🔍 UPI Fraud Detection — Interactive Dashboard")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Transactions")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head(200))

        # Basic stats
        st.markdown("**Basic dataset stats**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Unique users", df["user_id"].nunique())
        c3.metric("Avg. amount", f"{df['amount'].mean():.2f}")

        # Run predictions
        if st.button("Run Predictions"):
            st.info("Running predictions — this may take time for large files.")
            if mode == "Local (pickle)":
                if model is None or preprocessor is None:
                    st.error("Local model or preprocessor not found. Place fraud_model_xgb.pkl & preprocessor.pkl in app folder.")
                else:
                    probs, preds = predict_local(df, model, preprocessor)
                    df["fraud_probability"] = probs
                    df["is_fraud"] = preds
                    st.success("Predictions added (local).")
            else:
                # API mode: iterate rows (streamlit may timeout for huge files)
                results = []
                for _, row in df.iterrows():
                    prob, pred = predict_api_row(row, api_url)
                    results.append((prob, pred))
                probs = np.array([r[0] for r in results])
                preds = np.array([r[1] for r in results])
                df["fraud_probability"] = probs
                df["is_fraud"] = preds
                st.success("Predictions fetched from API.")

            # Show top suspicious
            st.subheader("Top suspicious transactions")
            top = df.sort_values("fraud_probability", ascending=False).head(20)
            st.dataframe(top)
            # Download predictions
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

            # Summary chart
            st.subheader("Fraud summary")
            fig, ax = plt.subplots()
            df["is_fraud"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xticklabels(["Non-fraud","Fraud"], rotation=0)
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Probability distribution
            st.subheader("Fraud probability distribution")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["fraud_probability"], bins=50)
            ax2.set_xlabel("fraud_probability")
            st.pyplot(fig2)

    else:
        st.info("Upload a CSV to begin. Use the example CSV to test.")

with col2:
    st.header("Quick Predict (single transaction)")
    st.markdown("Fill a single transaction and get instant prediction.")
    with st.form("single_txn"):
        t_id = st.text_input("transaction_id", "999999")
        u_id = st.text_input("user_id", "1111")
        amt = st.number_input("amount", value=2000.0, format="%.2f")
        hour = st.number_input("time_hour", min_value=0, max_value=23, value=12)
        loc_match = st.selectbox("location_match", [1,0], index=0)
        num_last24 = st.number_input("num_txn_last_24h", min_value=0, value=1)
        avg7 = st.number_input("avg_amount_last_7d", value=1200.0, format="%.2f")
        blacklisted = st.selectbox("is_blacklisted_merchant", [0,1], index=0)
        tx_type = st.selectbox("transaction_type", ["P2P","P2M","QR","Autopay"])
        mcat = st.selectbox("merchant_category", ["Food","Travel","Shopping","Bills","Other"])
        device = st.selectbox("device_type", ["Android","iOS"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        single_df = pd.DataFrame([{
            "transaction_id": int(t_id),
            "user_id": int(u_id),
            "amount": float(amt),
            "time_hour": int(hour),
            "location_match": int(loc_match),
            "num_txn_last_24h": int(num_last24),
            "avg_amount_last_7d": float(avg7),
            "is_blacklisted_merchant": int(blacklisted),
            "transaction_type": tx_type,
            "merchant_category": mcat,
            "device_type": device
        }])
        if mode == "Local (pickle)":
            if model is None or preprocessor is None:
                st.error("Local model or preprocessor not found.")
            else:
                prob, pred = predict_local(single_df, model, preprocessor)
                st.metric("Fraud probability", f"{prob[0]:.4f}")
                st.metric("Is fraud", str(int(pred[0])))
        else:
            prob, pred = predict_api_row(single_df.iloc[0], api_url)
            st.metric("Fraud probability", f"{prob:.4f}")
            st.metric("Is fraud", str(pred))

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.markdown("**Notes:** Model threshold is 0.5 by default. For production adjust threshold using precision/recall tradeoff. Use TLS for API in production.")

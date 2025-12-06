from fastapi import FastAPI
import uvicorn
import numpy as np
import pickle

# Load model and preprocessor
model = pickle.load(open("fraud_model_xgb.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "UPI Fraud Detection API Working!"}

@app.post("/predict")
def predict(data: dict):

    # Convert input JSON → array
    df = np.array([[
        data["transaction_id"],
        data["user_id"],
        data["amount"],
        data["time_hour"],
        data["location_match"],
        data["num_txn_last_24h"],
        data["avg_amount_last_7d"],
        data["is_blacklisted_merchant"],
        data["transaction_type"],
        data["merchant_category"],
        data["device_type"]
    ]], dtype=object)

    # Convert to DataFrame with correct column order
    import pandas as pd
    df = pd.DataFrame(df, columns=[
        "transaction_id","user_id","amount","time_hour",
        "location_match","num_txn_last_24h","avg_amount_last_7d",
        "is_blacklisted_merchant","transaction_type",
        "merchant_category","device_type"
    ])

    # Preprocess input
    X = preprocessor.transform(df)

    # Predict fraud probability
    prob = model.predict_proba(X)[0][1]
    prediction = int(prob > 0.5)

    return {
        "fraud_probability": float(prob),
        "is_fraud": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

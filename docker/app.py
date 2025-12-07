from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
FEATURE_FILE = os.getenv("FEATURE_FILE", "feature_columns.txt")

# 1) Load trained model
model = joblib.load(MODEL_PATH)

# 2) Load feature column names we logged from train.py
with open(FEATURE_FILE, "r") as f:
    feature_cols = [c for c in f.read().strip().split(",") if c]

@app.post("/predict")
def predict(features: Dict[str, Any]):
    """
    features: JSON object with keys = feature names used in training.
    """

    # Turn incoming dict into DataFrame
    df = pd.DataFrame([features])

    # Re-order / fill missing columns so it matches training layout
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Run model
    try:
        pred = int(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Map 0/1 to labels if you like
    label = "success" if pred == 0 else "failure"

    return {"prediction": int(pred), "label": label}

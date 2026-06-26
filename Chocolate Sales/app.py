"""
Chocolate Sales - Boxes_Shipped Prediction API
-----------------------------------------------
Run:
    pip install fastapi uvicorn scikit-learn pandas numpy
    uvicorn chocolate_app:app --reload

POST to http://127.0.0.1:8000/predict  (sample body at the bottom).
Interactive docs at http://127.0.0.1:8000/docs

NOTE: this file must sit in the same folder as chocolate_model.pkl
(or change MODEL_PATH to an absolute path).
"""

import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "chocolate_model.pkl"

# ---- load bundle ONCE at startup ----
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_order = bundle["feature_order"]   # exact column layout from training
log_target = bundle.get("log_target", False)

app = FastAPI(title="Chocolate Boxes Predictor")


# ---- request schema: the RAW order, as a human would describe it ----
# Order_Date is a string like "2022-12-11"; the categoricals are plain names.
class Order(BaseModel):
    Product: str
    Country: str
    Channel: str
    Salesperson: str
    Order_Date: str
    Discount_Pct: float
    Price_per_Box: float
    Marketing_Spend: float


CATEGORICAL_FIELDS = ["Product", "Country", "Channel", "Salesperson"]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: Order):
    raw = order.model_dump() if hasattr(order, "model_dump") else order.dict()

    # 1. parse the date and engineer the SAME features as training
    dt = pd.to_datetime(raw["Order_Date"], errors="coerce")
    if pd.isna(dt):
        return {"error": f"could not parse Order_Date: {raw['Order_Date']}"}

    row = {
        "Discount_Pct": raw["Discount_Pct"],
        "Price_per_Box": raw["Price_per_Box"],
        "Marketing_Spend": raw["Marketing_Spend"],
        "Order_Year": dt.year,
        "Order_Month": dt.month,
        "Order_Quarter": dt.quarter,
        "Order_Day": dt.day,
        "Order_Weekday": dt.weekday(),
    }

    # 2. engineered feature (same formula as training)
    row["Discounted_Price"] = raw["Price_per_Box"] * (1 - raw["Discount_Pct"] / 100)

    # 3. one-hot: set the matching dummy column to 1.
    #    Column names follow get_dummies' "Field_Value" convention.
    #    If the value is unseen (or is the drop_first baseline), the name
    #    simply won't exist in feature_order and reindex drops it -> all
    #    dummies for that field stay 0, i.e. treated as the baseline.
    for field in CATEGORICAL_FIELDS:
        dummy_col = f"{field}_{raw[field]}"
        row[dummy_col] = 1

    # 4. build one row, then force it into the EXACT training column layout.
    #    reindex adds any missing dummy columns as 0 and drops extras,
    #    guaranteeing the model sees the same features in the same order.
    X = pd.DataFrame([row]).reindex(columns=feature_order, fill_value=0)

    # 5. predict; invert the log transform if the model trained on log1p
    pred = model.predict(X)[0]
    if log_target:
        pred = np.expm1(pred)

    return {"predicted_boxes": round(float(pred))}


# ---------------------------------------------------------------------------
# Sample request body for Postman (POST http://127.0.0.1:8000/predict):
#
# {
#   "Product": "Truffle Gift Box",
#   "Country": "Australia",
#   "Channel": "Retail",
#   "Salesperson": "Arjun Mehta",
#   "Order_Date": "2022-12-11",
#   "Discount_Pct": 3.5,
#   "Price_per_Box": 13.72,
#   "Marketing_Spend": 202.03
# }
#
# The real Boxes_Shipped for this row was 71, so a correct API should
# predict somewhere near that.
# ---------------------------------------------------------------------------

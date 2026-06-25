import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from datetime import datetime

bundle = pickle.load(open("car_price_model.pkl", "rb"))

model = bundle["model"]
encoders = bundle["encoders"]
mileage_cap = bundle["mileage_cap"]
feature_order= bundle["feature_order"]

app = FastAPI(title = "Used Car Price Prediction")

class Car(BaseModel):
    brand: str
    model: str
    model_year: int
    milage: float
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str


def safe_encode(col, value):
    le = encoders[col]
    return int(le.transform([value])[0]) if value in le.classes_ else 0

@app.get('/')
def health():
    return {'status': 200}

@app.post('/predict')
def predict(car: Car):
    row = car.model_dump() if hasattr(car, "model_dump") else car.dict()
    row["model_year"] = datetime.now().year - row["model_year"]
    row["milage"] = min(row["milage"], mileage_cap)
    for col in encoders:
        row[col] = safe_encode(col, row[col])
    X = pd.DataFrame([row])[feature_order]
    pred = model.predict(X)[0]
    price = np.expm1(pred)
    return {'Predicted Price': round(price, 2)}
    

# Chocolate Sales — Boxes Shipped Prediction

Predicts `Boxes_Shipped` for a single chocolate order from its pre-order features
(product, country, channel, salesperson, date, discount, price, marketing spend).

## The headline: target leakage in the original notebook
A popular public notebook on this dataset reports R2 ~= 0.9997. That score is the
result of **target leakage**: the `Amount` column equals
`Boxes_Shipped * Price_per_Box * (1 - Discount_Pct/100)`, so it encodes the answer.
It also wouldn't be known at prediction time. This project proves the leakage
(the Amount-vs-formula difference is ~0 for typical rows), removes `Amount`, and
reports an honest model.

## Results (after removing leakage)
| model         | MAE (boxes) | RMSE (boxes) | R2 (log scale) |
|---------------|-------------|--------------|----------------|
| Linear        | ~34         | ~77          | ~0.79          |
| Random Forest | ~30         | ~68          | ~0.82          |

Top drivers: `Discounted_Price`, `Marketing_Spend`, plus specific products / regions.

## Other cleaning decisions
- Dropped rows with negative `Boxes_Shipped` (impossible values, <3% of data).
- Median-filled missing numeric columns; mode-filled missing dates.
- Engineered date parts (year/month/quarter/day/weekday) for seasonality.
- Trained on `log1p(target)` because the target skew is ~3.5.

## Files
- `Chocolate_Sales_Prediction.ipynb` — full pipeline, runnable top to bottom.
- `Chocolate_Sales.csv` — dataset.
- `app.py` — FastAPI inference service.
- `requirements.txt` — dependencies.

## Run the notebook
Open in Jupyter and Run All. It writes `chocolate_model.pkl` at the end.

## Run the API
    pip install -r requirements.txt
    uvicorn app:app --reload

POST to http://127.0.0.1:8000/predict   (interactive docs at /docs):

    {
      "Product": "Truffle Gift Box",
      "Country": "Australia",
      "Channel": "Retail",
      "Salesperson": "Arjun Mehta",
      "Order_Date": "2022-12-11",
      "Discount_Pct": 3.5,
      "Price_per_Box": 13.72,
      "Marketing_Spend": 202.03
    }

That row's real Boxes_Shipped was 71, so a correct API predicts near 71.

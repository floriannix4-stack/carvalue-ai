# 🚗 CarValue AI by Florian Nix

**AI-powered used car market intelligence — find good deals before anyone else.**

A Streamlit prototype built on historical car sales data to predict the fair market value of unsold cars and classify each listing as a **Good Deal**, **Potential Deal**, or **Risky / Overpaying** — using the model's own error margin (MAE) as the classification threshold.

---

## 📸 Overview

The prototype features five interactive tabs:

| Tab | Description |
|-----|-------------|
| 🌍 **Market Overview** | Interactive US map with 7,834 unsold cars colour-coded by deal quality |
| 🔎 **Deal Finder** | Filter inventory by manufacturer, car type, fuel, mileage, price, and deal score |
| 🔮 **Price Estimator** | Enter any car's details + asking price → get an instant AI valuation verdict |
| 📊 **Analytics** | Plotly charts: price distributions, value gaps by location, fuel type breakdowns |
| 🤖 **Model Performance** | Model metrics, GridSearchCV comparison, threshold logic, feature list |

---

## 🧠 How the Classification Works

The model predicts the **market-clearing (sold) price** of a car based on its attributes — excluding the dealer's asking price (`Price-$`), which is intentionally withheld so the AI evaluates it independently.

**Gap** = Asking Price − AI Predicted Value

| Verdict | Condition | Logic |
|---------|-----------|-------|
| 🟢 Good Deal | Gap ≤ 1.5 × MAE (~$1,124) | Within 1.5× the model's average error — competitively priced |
| 🟡 Potential Deal | Gap ≤ 2.0 × MAE (~$1,498) | Within 2× the error margin — negotiate using AI value as anchor |
| 🔴 Risky / Overpaying | Gap > 2.0 × MAE (~$1,498) | Seller asks significantly more than what comparable cars have sold for |

The model's **MAE of ~$749** is used as the calibration unit for thresholds, making the classification statistically grounded rather than arbitrary.

---

## 🗂️ Project Structure

```
carvalue-ai/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # Offline model training script (run once)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── used_car_sales.csv          # Dataset (10,000 cars: 2,166 sold + 7,834 unsold)
│   └── gz_2010_us_040_00_500k.json # US state boundaries GeoJSON for the map
│
└── model/
    ├── model.pkl                   # Trained GradientBoosting pipeline (joblib)
    ├── thresholds.pkl              # Calibrated Q25/Q75 thresholds
    ├── model_meta.json             # Training metadata (MAE, R², feature list)
    └── model_comparison.csv        # RandomForest vs GradientBoosting comparison
```

---

## 📦 Dataset

- **10,000 car listings** across 17 US locations
- **2,166 sold** (used for training — ground-truth market prices)
- **7,834 unsold** (holdout set — used for live predictions in the app)
- Features: manufacturer, model, car type, fuel type, gearbox, colour, location, year, mileage, engine power, seats, doors

---

## 🤖 Model Details

| Item | Detail |
|------|--------|
| **Algorithm** | GradientBoostingRegressor (won GridSearchCV vs RandomForest) |
| **Target** | Sold Price-$ (actual market-clearing price) |
| **Preprocessing** | OneHotEncoder for categoricals, passthrough for numerals |
| **Target transform** | Log1p / expm1 (stabilises variance across price range) |
| **Test MAE** | ~$749 (~10.9% of mean sold price) |
| **Test R²** | ~0.56 |
| **Train/Test split** | 80/20 on sold cars only |

> **Key design decision:** `Price-$` (the dealer's asking price) is deliberately excluded from features. Including it caused the model to memorise the listing price (MAE ≈ $0) rather than learn true market dynamics.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — model training (GradientBoosting, GridSearchCV)
- [Plotly](https://plotly.com/python/) — interactive charts
- [pydeck](https://deckgl.readthedocs.io/) — geospatial map
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data processing

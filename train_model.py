import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH       = "data/used_car_sales.csv"
MODEL_PATH      = "model/model.pkl"
THRESHOLDS_PATH = "model/thresholds.pkl"
RESULTS_PATH    = "model/model_comparison.csv"
META_PATH       = "model/model_meta.json"

CURRENT_YEAR = 2026

df = pd.read_csv(DATA_PATH)

for c in ["Manufactured Year", "Mileage-KM", "Engine Power-HP",
          "Number of Seats", "Number of Doors", "Price-$", "Sold Price-$"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"Total rows: {len(df):,}")

df_sold   = df[df["Sold Price-$"].notna() & (df["Sold Price-$"] > 0)].copy()
df_unsold = df[(df["Sold Price-$"].isna()) | (df["Sold Price-$"] <= 0)].copy()

print(f"Sold (training pool): {len(df_sold):,}")
print(f"Unsold (holdout):     {len(df_unsold):,}")

def add_features(d):
    d = d.copy()
    d["Car Age"]          = (CURRENT_YEAR - d["Manufactured Year"]).clip(lower=1)
    d["Mileage per Year"] = d["Mileage-KM"] / d["Car Age"]
    return d

df_sold   = add_features(df_sold)
df_unsold = add_features(df_unsold)

needed = ["Manufacturer Name", "Car Name", "Car Type", "Color", "Gearbox",
          "Energy", "Location", "Manufactured Year", "Mileage-KM",
          "Engine Power-HP", "Number of Seats", "Number of Doors", "Sold Price-$"]
df_sold = df_sold.dropna(subset=needed).copy()

needed_unsold = [c for c in needed if c != "Sold Price-$"] + ["Price-$"]
df_unsold = df_unsold.dropna(subset=needed_unsold).copy()

cat_cols = ["Manufacturer Name", "Car Name", "Car Type",
            "Color", "Gearbox", "Energy", "Location"]
num_cols = ["Car Age", "Mileage-KM", "Mileage per Year",
            "Engine Power-HP", "Number of Seats", "Number of Doors"]
FEATURES = cat_cols + num_cols

X = df_sold[FEATURES]
y = df_sold["Sold Price-$"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)


def make_rf():
    return Pipeline([
        ("prep", preprocessor),
        ("regressor", TransformedTargetRegressor(
            regressor=RandomForestRegressor(random_state=42, n_jobs=-1),
            func=np.log1p, inverse_func=np.expm1,
        )),
    ])

def make_gbr():
    return Pipeline([
        ("prep", preprocessor),
        ("regressor", TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(random_state=42),
            func=np.log1p, inverse_func=np.expm1,
        )),
    ])

candidates = [
    ("RandomForest", make_rf(), {
        "regressor__regressor__n_estimators":  [300, 600],
        "regressor__regressor__max_depth":     [10, 15, None],
        "regressor__regressor__min_samples_leaf": [1, 2, 4],
    }),
    ("GradientBoosting", make_gbr(), {
        "regressor__regressor__n_estimators":  [200, 400],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__regressor__max_depth":     [2, 3],
    }),
]

results   = []
best_model, best_name, best_mae = None, None, float("inf")

for name, pipe, grid in candidates:
    print(f"\n=== Tuning: {name} ===")
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                      cv=5, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    m = gs.best_estimator_
    preds = m.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"  Best params : {gs.best_params_}")
    print(f"  MAE  : ${mae:,.0f}  ({mae/y_test.mean():.1%} of mean)")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.3f}")

    results.append({"model": name, "cv_mae": float(-gs.best_score_),
                    "test_mae": float(mae), "test_rmse": float(rmse),
                    "test_r2": float(r2), "best_params": json.dumps(gs.best_params_)})

    if mae < best_mae:
        best_mae, best_model, best_name = mae, m, name


unsold_preds = best_model.predict(df_unsold[FEATURES])
value_pct    = (unsold_preds - df_unsold["Price-$"].values) / unsold_preds

threshold_low  = float(np.percentile(value_pct, 25))
threshold_high = float(np.percentile(value_pct, 75))

print(f"\n=== Threshold calibration ({len(df_unsold):,} unsold cars) ===")
print(f"  Value% mean : {value_pct.mean():.3f}")
print(f"  Value% std  : {value_pct.std():.3f}")
print(f"  Overpriced  ≤ Q25 : {threshold_low:.3f}")
print(f"  Undervalued ≥ Q75 : {threshold_high:.3f}")
print(f"  → Undervalued : {(value_pct >= threshold_high).mean():.1%}")
print(f"  → Fair Value  : {((value_pct > threshold_low) & (value_pct < threshold_high)).mean():.1%}")
print(f"  → Overpriced  : {(value_pct <= threshold_low).mean():.1%}")

pd.DataFrame(results).sort_values("test_mae").to_csv(RESULTS_PATH, index=False)

joblib.dump(best_model, MODEL_PATH)
joblib.dump({"threshold_low": threshold_low, "threshold_high": threshold_high},
            THRESHOLDS_PATH)

with open(META_PATH, "w") as f:
    json.dump({"best_model": best_name, "best_test_mae": float(best_mae),
               "features": FEATURES, "train_rows": int(len(X_train)),
               "test_rows": int(len(X_test)),
               "threshold_low": threshold_low,
               "threshold_high": threshold_high}, f, indent=2)

print(f"\n✓ model.pkl        → {MODEL_PATH}")
print(f"✓ thresholds.pkl   → {THRESHOLDS_PATH}")
print(f"✓ model_comparison → {RESULTS_PATH}")
print(f"✓ model_meta       → {META_PATH}")
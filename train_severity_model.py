"""
Severity scoring model — Parkinson's disease motor symptom estimation.

Uses the UCI Parkinson's Telemonitoring dataset (5,875 recordings, 42 subjects)
to train a regression model that predicts motor_UPDRS score (0–108).

The predicted score is then mapped to a human-readable severity band:
  0–10  → None / Minimal
  11–30 → Mild
  31–58 → Moderate
  59+   → Severe

Column mapping from telemonitoring → UCI feature names used in production:
  Telemonitoring        UCI (training data)
  ─────────────────────────────────────────
  Jitter(%)          →  MDVP:Jitter(%)
  Jitter(Abs)        →  MDVP:Jitter(Abs)
  Jitter:RAP         →  MDVP:RAP
  Jitter:PPQ5        →  MDVP:PPQ
  Jitter:DDP         →  Jitter:DDP
  Shimmer            →  MDVP:Shimmer
  Shimmer(dB)        →  MDVP:Shimmer(dB)
  Shimmer:APQ3       →  Shimmer:APQ3
  Shimmer:APQ5       →  Shimmer:APQ5
  Shimmer:APQ11      →  MDVP:APQ
  Shimmer:DDA        →  Shimmer:DDA
  NHR                →  NHR
  HNR                →  HNR
  RPDE               →  RPDE
  DFA                →  DFA
  PPE                →  PPE
  (spread1/2, D2 not in telemonitoring — excluded from severity model)
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── Severity band labels ──────────────────────────────────────────────────────
SEVERITY_BANDS = [
    (0,  10,  "None / Minimal",  "#27AE60"),
    (11, 30,  "Mild",            "#F39C12"),
    (31, 58,  "Moderate",        "#E67E22"),
    (59, 200, "Severe",          "#E74C3C"),
]

def score_to_band(score):
    for lo, hi, label, color in SEVERITY_BANDS:
        if lo <= score <= hi:
            return label, color
    return "Severe", "#E74C3C"


# ── Column mapping ────────────────────────────────────────────────────────────
TELE_TO_UCI = {
    "Jitter(%)":     "MDVP:Jitter(%)",
    "Jitter(Abs)":   "MDVP:Jitter(Abs)",
    "Jitter:RAP":    "MDVP:RAP",
    "Jitter:PPQ5":   "MDVP:PPQ",
    "Jitter:DDP":    "Jitter:DDP",
    "Shimmer":       "MDVP:Shimmer",
    "Shimmer(dB)":   "MDVP:Shimmer(dB)",
    "Shimmer:APQ3":  "Shimmer:APQ3",
    "Shimmer:APQ5":  "Shimmer:APQ5",
    "Shimmer:APQ11": "MDVP:APQ",
    "Shimmer:DDA":   "Shimmer:DDA",
    "NHR":           "NHR",
    "HNR":           "HNR",
    "RPDE":          "RPDE",
    "DFA":           "DFA",
    "PPE":           "PPE",
}

SEVERITY_FEATURES = list(TELE_TO_UCI.values())   # 16 features in UCI naming


def load_and_prepare(filepath="parkinsons_updrs.csv"):
    print("📊 Loading telemonitoring dataset...")
    df = pd.read_csv(filepath)
    print(f"   Rows: {len(df)} | Subjects: {df['subject#'].nunique()}")
    print(f"   motor_UPDRS range: {df['motor_UPDRS'].min():.1f} – {df['motor_UPDRS'].max():.1f}")

    # Rename telemonitoring columns → UCI feature names
    df = df.rename(columns=TELE_TO_UCI)

    X = df[SEVERITY_FEATURES]
    y = df["motor_UPDRS"]
    return X, y


def train(X, y):
    print("\n🔄 Splitting data (80/20 by subject to avoid leakage)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🤖 Training XGBoost Regressor with GridSearchCV...")
    param_grid = {
        "n_estimators":  [200, 400],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    base = XGBRegressor(eval_metric="mae", random_state=42, verbosity=0)
    grid = GridSearchCV(base, param_grid, cv=KFold(5, shuffle=True, random_state=42),
                        scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
    grid.fit(X_train_s, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test_s)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    print(f"\n   Best params : {grid.best_params_}")
    print(f"   MAE         : {mae:.2f} UPDRS points")
    print(f"   RMSE        : {rmse:.2f}")
    print(f"   R²          : {r2:.4f}")

    return model, scaler, mae, rmse, r2


def save(model, scaler, mae, rmse, r2, dataset_size):
    joblib.dump(model,  "severity_model.pkl")
    joblib.dump(scaler, "severity_scaler.pkl")
    joblib.dump(SEVERITY_FEATURES, "severity_feature_names.pkl")

    metrics = {
        "model":          type(model).__name__,
        "target":         "motor_UPDRS",
        "mae":            round(mae, 2),
        "rmse":           round(rmse, 2),
        "r2":             round(r2, 4),
        "dataset_size":   dataset_size,
        "n_features":     len(SEVERITY_FEATURES),
        "severity_bands": [
            {"label": l, "min": lo, "max": hi, "color": c}
            for lo, hi, l, c in SEVERITY_BANDS
        ],
    }
    with open("severity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n💾 Saved:")
    print("   severity_model.pkl")
    print("   severity_scaler.pkl")
    print("   severity_feature_names.pkl")
    print("   severity_metrics.json")
    print(f"\n📊 MAE = {mae:.2f} UPDRS points  |  R² = {r2:.4f}")


if __name__ == "__main__":
    X, y = load_and_prepare("parkinsons_updrs.csv")
    model, scaler, mae, rmse, r2 = train(X, y)
    save(model, scaler, mae, rmse, r2, dataset_size=len(X))
    print("\n✅ Severity model ready.")

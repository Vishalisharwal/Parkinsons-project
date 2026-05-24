"""
Main classification pipeline — Parkinson's disease binary detection.

Dataset:
  - UCI Parkinson's Voice Dataset        (195 samples, 22 features, 147 PD / 48 Healthy)
  - UCI Parkinson's Telemonitoring Dataset (5,875 samples, 16 features, ALL PD patients)

Combined strategy:
  - All telemonitoring rows are labeled PD=1 (all 42 subjects have diagnosed PD).
  - The intersection of both datasets is 16 features (F0/Fhi/Flo and spread1/2/D2 only
    exist in the UCI dataset — they are dropped when combining).
  - SMOTE upsamples the 48 healthy samples to ~25% of the PD class (≈1,500 synthetic
    healthy samples) to avoid extreme imbalance without over-generating.
  - Final training set: ~6,022 PD + ~1,500 healthy = ~7,500 balanced samples.

The benefit: the model sees 42 different PD patients across varying severity levels
and time points, making it far more robust than training on 195 samples alone.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")


# Telemonitoring column names → UCI column names
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

# 16 features present in both datasets (after renaming)
COMMON_FEATURES = list(TELE_TO_UCI.values())


class ParkinsonsPipeline:
    """
    Combined ML pipeline for Parkinson's disease detection.
    Trains on UCI + Telemonitoring datasets for improved robustness.
    """

    def __init__(self):
        self.scaler          = StandardScaler()
        self.best_model      = None
        self.best_model_name = None
        self.feature_names   = None

        self.model_configs = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=42),
                {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
                 "min_samples_split": [2, 5]}
            ),
            "SVM": (
                SVC(probability=True, random_state=42),
                {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}
            ),
            "XGBoost": (
                XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
                {"n_estimators": [100, 200], "max_depth": [3, 5],
                 "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}
            ),
        }

    def load_and_prepare_data(self,
                               uci_path="parkinsons.csv",
                               tele_path="parkinsons_updrs.csv"):
        """
        Load both datasets and combine on the 16 common features.

        UCI dataset  : 195 rows, original binary status label
        Telemonitoring: 5,875 rows, ALL labeled PD=1 (no healthy controls)
        """
        print("📊 Loading datasets...")

        # ── UCI dataset ───────────────────────────────────────────────────────
        uci = pd.read_csv(uci_path)
        uci_X = uci[COMMON_FEATURES]
        uci_y = uci["status"]

        uci_pd      = (uci_y == 1).sum()
        uci_healthy = (uci_y == 0).sum()
        print(f"   UCI dataset      : {len(uci)} rows  |  PD: {uci_pd}  Healthy: {uci_healthy}")

        # ── Telemonitoring dataset ────────────────────────────────────────────
        tele = pd.read_csv(tele_path)
        tele = tele.rename(columns=TELE_TO_UCI)
        tele_X = tele[COMMON_FEATURES]
        tele_y = pd.Series(np.ones(len(tele), dtype=int),
                           name="status",
                           index=tele.index)
        print(f"   Telemonitoring   : {len(tele)} rows  |  PD: {len(tele)}  Healthy: 0")
        print(f"   (All 42 subjects have diagnosed Parkinson's — no healthy controls)")

        # ── Combine ───────────────────────────────────────────────────────────
        X = pd.concat([uci_X, tele_X], ignore_index=True)
        y = pd.concat([uci_y, tele_y], ignore_index=True)

        total_pd      = (y == 1).sum()
        total_healthy = (y == 0).sum()
        print(f"\n   Combined total   : {len(X)} rows  |  PD: {total_pd}  Healthy: {total_healthy}")
        print(f"   Features used    : {len(COMMON_FEATURES)} (16 common features)")
        print(f"   Dropped features : MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), spread1, spread2, D2")

        self.feature_names = COMMON_FEATURES
        return X, y

    def train_and_evaluate(self, X, y, test_size=0.2):
        """
        Full training pipeline with SMOTE and GridSearchCV.

        SMOTE strategy: upsample healthy class to 25% of PD class size.
        This avoids the 125:1 extreme ratio while keeping synthetic healthy
        samples numerically reasonable relative to the 48 real healthy recordings.
        """
        print("\n🔄 Splitting data (stratified 80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print("⚖️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # SMOTE: upsample healthy to 25% of PD count
        pd_count      = (y_train == 1).sum()
        target_healthy = int(pd_count * 0.25)
        print(f"🔁 Applying SMOTE — upsampling Healthy from "
              f"{(y_train==0).sum()} → {target_healthy} "
              f"(25% of PD class = {pd_count})")

        smote = SMOTE(sampling_strategy={0: target_healthy}, random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        print(f"   After SMOTE — PD: {(y_train_bal==1).sum()}  Healthy: {(y_train_bal==0).sum()}")
        print(f"\n   Train set: {len(X_train_bal)} samples | Test set: {len(X_test)} samples\n")

        results      = {}
        best_roc_auc = 0.0
        cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, (base_model, param_grid) in self.model_configs.items():
            print(f"🤖 Training {name} with GridSearchCV...")

            grid = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring="roc_auc",
                n_jobs=-1, refit=True
            )
            grid.fit(X_train_bal, y_train_bal)
            tuned = grid.best_estimator_

            y_pred      = tuned.predict(X_test_scaled)
            y_pred_prob = tuned.predict_proba(X_test_scaled)[:, 1]

            acc     = accuracy_score(y_test, y_pred)
            prec    = precision_score(y_test, y_pred, zero_division=0)
            rec     = recall_score(y_test, y_pred, zero_division=0)
            f1      = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            cm      = confusion_matrix(y_test, y_pred).tolist()

            results[name] = {
                "model":            tuned,
                "accuracy":         acc,
                "precision":        prec,
                "recall":           rec,
                "f1":               f1,
                "roc_auc":          roc_auc,
                "confusion_matrix": cm,
                "best_params":      grid.best_params_,
                "y_pred":           y_pred,
                "y_test":           y_test,
            }

            print(f"   Best params : {grid.best_params_}")
            print(f"   Accuracy    : {acc:.4f}")
            print(f"   Precision   : {prec:.4f}")
            print(f"   Recall      : {rec:.4f}")
            print(f"   F1-Score    : {f1:.4f}")
            print(f"   ROC-AUC     : {roc_auc:.4f}\n")

            if roc_auc > best_roc_auc:
                best_roc_auc         = roc_auc
                self.best_model      = tuned
                self.best_model_name = name

        print(f"🏆 Best Model: {self.best_model_name}  (ROC-AUC: {best_roc_auc:.4f})\n")
        self._display_best_model_results(results[self.best_model_name])
        return results

    def _display_best_model_results(self, result):
        print(f"📈 Detailed Results — {self.best_model_name}:")
        print(classification_report(
            result["y_test"], result["y_pred"],
            target_names=["Healthy", "Parkinson's"]
        ))
        cm = result["confusion_matrix"]
        print("Confusion Matrix:")
        print(f"  True Negatives : {cm[0][0]}  |  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}  |  True Positives : {cm[1][1]}")

    def save_pipeline(self, model_path="best_model.pkl", scaler_path="scaler.pkl",
                      metrics_path="metrics.json"):
        if self.best_model is None:
            print("❌ No model trained yet!")
            return

        joblib.dump(self.best_model,    model_path)
        joblib.dump(self.scaler,        scaler_path)
        joblib.dump(self.feature_names, "feature_names.pkl")

        # Evaluate on full UCI dataset for the saved metrics
        uci = pd.read_csv("parkinsons.csv")
        X_uci = uci[COMMON_FEATURES]
        y_uci = uci["status"]
        X_scaled = self.scaler.transform(X_uci)
        y_pred   = self.best_model.predict(X_scaled)
        y_prob   = self.best_model.predict_proba(X_scaled)[:, 1]

        # Also count combined dataset size for reporting
        tele = pd.read_csv("parkinsons_updrs.csv")
        total_samples = len(uci) + len(tele)

        metrics = {
            "model_name":        type(self.best_model).__name__,
            "model_label":       self.best_model_name,
            "accuracy":          round(accuracy_score(y_uci, y_pred), 4),
            "precision":         round(precision_score(y_uci, y_pred, zero_division=0), 4),
            "recall":            round(recall_score(y_uci, y_pred, zero_division=0), 4),
            "f1":                round(f1_score(y_uci, y_pred, zero_division=0), 4),
            "roc_auc":           round(roc_auc_score(y_uci, y_prob), 4),
            "confusion_matrix":  confusion_matrix(y_uci, y_pred).tolist(),
            "dataset_size":      total_samples,
            "dataset_note":      "UCI (195) + Telemonitoring (5,875) — 16 common features",
            "n_features":        len(self.feature_names),
            "cv_folds":          5,
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Pipeline saved:")
        print(f"   Model    → {model_path}")
        print(f"   Scaler   → {scaler_path}")
        print(f"   Features → feature_names.pkl  ({len(self.feature_names)} features)")
        print(f"   Metrics  → {metrics_path}")
        print(f"\n📊 Evaluation on UCI test set (195 samples):")
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "dataset_note"):
                print(f"   {k}: {v}")

    def predict(self, features):
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        if isinstance(features, dict):
            features = [features.get(name, 0) for name in self.feature_names]
        features_array  = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        prediction  = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0]
        return {
            "has_parkinsons":         bool(prediction),
            "probability_healthy":    probability[0],
            "probability_parkinsons": probability[1],
            "confidence":             max(probability),
            "model_used":             self.best_model_name,
        }

    @staticmethod
    def load_pipeline(model_path="best_model.pkl", scaler_path="scaler.pkl"):
        pipeline = ParkinsonsPipeline()
        pipeline.best_model    = joblib.load(model_path)
        pipeline.scaler        = joblib.load(scaler_path)
        pipeline.feature_names = joblib.load("feature_names.pkl")
        pipeline.best_model_name = type(pipeline.best_model).__name__
        return pipeline


# ── Run training ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = ParkinsonsPipeline()
    X, y     = pipeline.load_and_prepare_data()
    results  = pipeline.train_and_evaluate(X, y)
    pipeline.save_pipeline()

    print("\n" + "=" * 60)
    print("🎯 PREDICTION EXAMPLE (first UCI row)")
    print("=" * 60)
    uci    = pd.read_csv("parkinsons.csv")
    sample = uci[COMMON_FEATURES].iloc[0].values
    result = pipeline.predict(sample)
    print(f"  Status      : {'Parkinson\'s' if result['has_parkinsons'] else 'Healthy'}")
    print(f"  PD Prob     : {result['probability_parkinsons']:.2%}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print(f"  Model Used  : {result['model_used']}")
    print("\n✅ Pipeline ready!  Run: streamlit run streamlit_app.py")

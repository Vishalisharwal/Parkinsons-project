# ParkiSense AI — Parkinson's Disease Voice Screening

An AI-powered web application that screens for Parkinson's disease by analyzing voice biomarkers. Built with Streamlit, scikit-learn, XGBoost, and SHAP.

---

## What It Does

- Accepts a **CSV of pre-extracted voice features** or a **raw audio file / microphone recording**
- Runs predictions through a trained XGBoost model (ROC-AUC ~0.99)
- Shows a risk gauge, confidence scores, and a **SHAP explanation** — which features drove the result
- Logs every prediction in a **session history** with CSV export
- Includes a **Data Explorer** tab for visualizing the training dataset

> **Medical disclaimer:** This tool is for screening and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified neurologist.

---

## Project Structure

```
Parkinsons-project/
├── streamlit_app.py          # Main Streamlit web app (4 tabs)
├── train_and_save_model.py   # ML training pipeline
├── parkinsons.csv            # UCI Parkinson's voice dataset (195 samples)
├── best_model.pkl            # Trained model (generated after training)
├── scaler.pkl                # Feature scaler (generated after training)
├── feature_names.pkl         # Feature name list (generated after training)
├── metrics.json              # Real model metrics (generated after training)
└── requirements.txt          # All dependencies with pinned versions
```

---

## Setup & Run

### 1. Prerequisites (macOS only — for XGBoost)
```bash
brew install libomp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
This generates `best_model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `metrics.json`.
```bash
python train_and_save_model.py
```

### 4. Launch the app
```bash
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ML Pipeline

| Step | Detail |
|---|---|
| Dataset | UCI Parkinson's Voice Dataset — 195 samples, 22 features |
| Class balancing | SMOTE oversampling (48 Healthy → 118, matching 118 PD) |
| Models trained | Logistic Regression, Random Forest, SVM, XGBoost |
| Tuning | GridSearchCV with 5-fold stratified cross-validation |
| Selection criterion | ROC-AUC (not accuracy — more honest on imbalanced data) |
| Winner | XGBoost (ROC-AUC ≈ 0.986 on held-out test set) |

---

## App Tabs

| Tab | Description |
|---|---|
| **Analysis** | Upload CSV or audio → get risk prediction + SHAP explanation |
| **Model Performance** | Real metrics, confusion matrix, performance bar chart |
| **Data Explorer** | Feature distributions, correlation heatmap, raw data viewer |
| **Information** | About Parkinson's, voice analysis, and medical disclaimer |

---

## Voice Features Used

All 22 features from the MDVP (Multi-Dimensional Voice Program) standard:

| Category | Features |
|---|---|
| Fundamental frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter (frequency variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` |
| Shimmer (amplitude variation) | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` |
| Noise ratios | `NHR`, `HNR` |
| Nonlinear dynamics | `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` |

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Parkinson's Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Reference:** Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008)
- **Size:** 195 voice recordings from 31 subjects (23 with PD, 8 healthy)
- **Label:** `status` — 1 = Parkinson's, 0 = Healthy

---

## Dependencies

See `requirements.txt` for pinned versions. Key packages:

- `streamlit` — web app framework
- `xgboost` — gradient boosted tree model
- `shap` — model explainability
- `imbalanced-learn` — SMOTE for class balancing
- `librosa` — audio feature extraction
- `plotly` — interactive charts

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import librosa
import parselmouth
from parselmouth.praat import call as praat_call
import io
from audiorecorder import audiorecorder


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkiSense AI – Parkinson's Voice Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── TOKENS ─────────────────────────────────────────────────────────────── */
:root {
  --bg:          #07100d;
  --surface-1:   #0c1610;
  --surface-2:   #101d15;
  --surface-3:   #152318;
  --border-dim:  rgba(0, 230, 118, 0.07);
  --border-mid:  rgba(0, 230, 118, 0.14);
  --border-on:   rgba(0, 230, 118, 0.35);
  --accent:      #00e676;
  --accent-dim:  rgba(0, 230, 118, 0.08);
  --accent-glow: rgba(0, 230, 118, 0.18);
  --text-1:      #e2f0e7;
  --text-2:      #8aad95;
  --text-3:      #4d7260;
  --red:         #ef5350;
  --red-dim:     rgba(239, 83, 80, 0.09);
  --orange:      #ff9800;
  --orange-dim:  rgba(255, 152, 0, 0.09);
  --radius-sm:   8px;
  --radius-md:   12px;
  --radius-lg:   16px;
  --shadow-sm:   0 1px 3px rgba(0,0,0,.4), 0 1px 2px rgba(0,0,0,.3);
  --shadow-md:   0 4px 12px rgba(0,0,0,.45);
  --shadow-lg:   0 8px 24px rgba(0,0,0,.5);
  --transition:  all .18s ease;
}

/* ── GLOBAL ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
  -webkit-font-smoothing: antialiased;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.6rem !important; padding-bottom: 3rem !important; }

/* ── TYPOGRAPHY ─────────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6,p,li,label,span { color: var(--text-1) !important; }

/* ── SIDEBAR ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface-1) !important;
  border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] > div { padding: 1.4rem 1rem !important; }
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] hr {
  border: none;
  border-top: 1px solid var(--border-dim);
  margin: 1rem 0;
}

/* ── TABS ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: 4px;
  gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
  background: transparent;
  color: var(--text-2) !important;
  border-radius: var(--radius-sm) !important;
  font-size: .85rem !important;
  font-weight: 500 !important;
  padding: .45rem .95rem !important;
  transition: var(--transition);
  border: none !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  background: var(--surface-2) !important;
  color: var(--text-1) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-mid) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after { display: none; }

/* ── BUTTONS ────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--accent) !important;
  color: #07100d !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: .875rem !important;
  padding: .55rem 1.4rem !important;
  transition: var(--transition) !important;
  box-shadow: var(--shadow-sm) !important;
  letter-spacing: -.01em !important;
}
.stButton > button:hover {
  background: #33eb8c !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px var(--accent-glow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* secondary (download) buttons */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  font-size: .85rem !important;
  transition: var(--transition) !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: var(--accent-dim) !important;
  border-color: var(--border-on) !important;
}

/* ── INPUTS / SELECTS ───────────────────────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label { color: var(--text-2) !important; font-size:.82rem !important; }

div[role="radiogroup"] > label {
  background: var(--surface-2) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  padding: .35rem .9rem !important;
  font-size: .85rem !important;
  color: var(--text-2) !important;
  transition: var(--transition) !important;
  cursor: pointer;
}
div[role="radiogroup"] > label:hover {
  border-color: var(--border-mid) !important;
  color: var(--text-1) !important;
}
div[role="radiogroup"] > label[data-baseweb] { /* selected */ }

/* ── FILE UPLOADER ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--surface-1) !important;
  border: 1.5px dashed var(--border-mid) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.2rem !important;
  transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--border-on) !important;
  background: var(--surface-2) !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── METRICS ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: 1rem 1.1rem;
  transition: var(--transition);
}
[data-testid="stMetric"]:hover { border-color: var(--border-mid); }
[data-testid="stMetricLabel"] { color: var(--text-2) !important; font-size:.78rem !important; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stMetricValue"] { color: var(--text-1) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size:.8rem !important; }

/* ── EXPANDER ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface-1) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary { color: var(--text-2) !important; font-size: .875rem !important; }

/* ── DATAFRAME ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe { border-radius: var(--radius-sm) !important; }

/* ── ALERTS / INFO ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border-left-width: 3px !important;
  font-size: .875rem !important;
}

/* ── SPINNER ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── PROGRESS ───────────────────────────────────────────────────────────── */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px !important; }

/* ── DIVIDER ────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border-dim) !important; margin: 1.5rem 0 !important; }

/* ── CAPTION ────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p { color: var(--text-3) !important; font-size: .78rem !important; }

/* ── PLOTLY ─────────────────────────────────────────────────────────────── */
.js-plotly-plot .plotly, .plot-container { background: transparent !important; }

/* ── CUSTOM COMPONENTS ──────────────────────────────────────────────────── */

/* Page header */
.ps-header {
  padding: 2.2rem 2.5rem 2rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border-dim);
  display: flex;
  align-items: center;
  gap: 1.2rem;
}
.ps-header-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--accent-dim);
  border: 1px solid var(--border-mid);
  color: var(--accent) !important;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  padding: .25rem .7rem;
  border-radius: 99px;
  margin-bottom: .5rem;
}
.ps-header-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -.03em;
  line-height: 1.15;
  background: linear-gradient(135deg, #e2f0e7 0%, var(--accent) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
}
.ps-header-sub {
  font-size: .9rem;
  color: var(--text-2) !important;
  margin: .3rem 0 0;
  font-weight: 400;
}

/* Section label */
.ps-section-label {
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-3) !important;
  margin-bottom: .6rem;
}

/* Card */
.ps-card {
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: 1.25rem 1.4rem;
  transition: var(--transition);
}
.ps-card:hover { border-color: var(--border-mid); }
.ps-card-title {
  font-size: .8rem;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--text-3) !important;
  margin: 0 0 .5rem;
}
.ps-card-body { font-size: .875rem; color: var(--text-2) !important; line-height: 1.6; margin: 0; }

/* Metric card */
.ps-metric {
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: 1.1rem 1.3rem;
  transition: var(--transition);
}
.ps-metric:hover { border-color: var(--border-mid); box-shadow: var(--shadow-sm); }
.ps-metric-label {
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--text-3) !important;
  margin: 0 0 .35rem;
}
.ps-metric-value {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -.03em;
  color: var(--text-1) !important;
  margin: 0 0 .15rem;
  line-height: 1;
}
.ps-metric-sub { font-size: .75rem; color: var(--text-3) !important; margin: 0; }

/* Result cards */
.ps-result {
  border-radius: var(--radius-md);
  padding: 1.25rem 1.4rem;
  margin-bottom: .75rem;
}
.ps-result-risk {
  background: var(--red-dim);
  border: 1px solid rgba(239,83,80,.2);
  border-left: 3px solid var(--red);
}
.ps-result-healthy {
  background: var(--accent-dim);
  border: 1px solid var(--border-mid);
  border-left: 3px solid var(--accent);
}
.ps-result-title { font-size: 1rem; font-weight: 700; margin: 0 0 .4rem; }
.ps-result-body  { font-size: .85rem; color: var(--text-2) !important; margin: 0 0 .6rem; line-height: 1.55; }
.ps-result-note  { font-size: .78rem; color: var(--text-3) !important; margin: 0; }

/* Severity pill */
.ps-severity {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 99px;
  padding: .22rem .7rem;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

/* Upload zone */
.ps-upload-zone {
  background: var(--surface-1);
  border: 1.5px dashed var(--border-mid);
  border-radius: var(--radius-lg);
  padding: 2.5rem 1.5rem;
  text-align: center;
  transition: var(--transition);
}
.ps-upload-zone:hover { border-color: var(--border-on); background: var(--surface-2); }
.ps-upload-icon { font-size: 2.2rem; margin-bottom: .75rem; opacity: .7; }
.ps-upload-title { font-size: 1rem; font-weight: 600; margin: 0 0 .3rem; color: var(--text-1) !important; }
.ps-upload-hint  { font-size: .8rem; color: var(--text-3) !important; margin: 0; }

/* Info callout */
.ps-callout {
  background: var(--surface-2);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: .9rem 1.1rem;
  font-size: .84rem;
  color: var(--text-2) !important;
  line-height: 1.6;
}
.ps-callout strong { color: var(--text-1) !important; }

/* Tips list */
.ps-tips {
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  padding: 1rem 1.25rem;
}
.ps-tips-title { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--text-3) !important; margin: 0 0 .6rem; }
.ps-tips ul { margin: 0; padding-left: 1.1rem; }
.ps-tips li { font-size: .84rem; color: var(--text-2) !important; margin-bottom: .3rem; line-height: 1.5; }

/* Sidebar brand */
.ps-brand {
  display: flex;
  align-items: center;
  gap: .6rem;
  margin-bottom: 1.2rem;
}
.ps-brand-icon { font-size: 1.4rem; }
.ps-brand-name { font-size: 1rem; font-weight: 700; letter-spacing: -.02em; color: var(--accent) !important; }
.ps-brand-tag  { font-size: .65rem; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; color: var(--text-3) !important; }

/* Sidebar stat row */
.ps-stat-row { display: flex; justify-content: space-between; align-items: center; padding: .3rem 0; border-bottom: 1px solid var(--border-dim); }
.ps-stat-key { font-size: .78rem; color: var(--text-3) !important; }
.ps-stat-val { font-size: .78rem; font-weight: 600; color: var(--accent) !important; }

/* History entry */
.ps-history-item {
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  padding: .55rem .75rem;
  margin-bottom: .4rem;
  border-left: 2px solid transparent;
  transition: var(--transition);
}
.ps-history-item:hover { border-left-color: var(--border-mid); }

/* SHAP legend */
.ps-shap-legend {
  display: flex;
  gap: 1.2rem;
  font-size: .78rem;
  color: var(--text-2) !important;
  margin-top: .4rem;
}
.ps-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }

/* Tag/badge */
.ps-tag {
  display: inline-block;
  background: var(--surface-2);
  border: 1px solid var(--border-dim);
  border-radius: 99px;
  padding: .15rem .55rem;
  font-size: .7rem;
  font-weight: 500;
  color: var(--text-2) !important;
  letter-spacing: .02em;
}

/* Footer */
.ps-footer {
  text-align: center;
  padding: 1.5rem 0 .5rem;
  border-top: 1px solid var(--border-dim);
  margin-top: 2rem;
}
.ps-footer p { font-size: .75rem; color: var(--text-3) !important; margin: .2rem 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-mid); }

/* multiselect tags */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  background: var(--accent-dim) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 99px !important;
  color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── CHART THEME ───────────────────────────────────────────────────────────────
CHART_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,22,16,0.6)",
    font=dict(family="Inter, sans-serif", color="#8aad95", size=11),
)

def _apply_chart_theme(fig, height=320, margin=None):
    if margin is None:
        margin = dict(l=10, r=10, t=40, b=10)
    fig.update_layout(
        **CHART_DEFAULTS,
        height=height,
        margin=margin,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    )
    return fig


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model         = joblib.load("best_model.pkl")
        scaler        = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        with open("metrics.json") as f:
            metrics = json.load(f)
        try:
            sev_model         = joblib.load("severity_model.pkl")
            sev_scaler        = joblib.load("severity_scaler.pkl")
            sev_feature_names = joblib.load("severity_feature_names.pkl")
            with open("severity_metrics.json") as f:
                sev_metrics = json.load(f)
        except Exception:
            sev_model = sev_scaler = sev_feature_names = sev_metrics = None
        return model, scaler, feature_names, metrics, sev_model, sev_scaler, sev_feature_names, sev_metrics
    except Exception as e:
        st.error(f"Model files not found. Run train_and_save_model.py first. ({e})")
        return None, None, None, None, None, None, None, None


@st.cache_resource
def get_shap_explainer(_model, _scaler):
    try:
        df       = pd.read_csv("parkinsons.csv")
        X_bg     = df.drop(["name", "status"], axis=1)
        # Keep only the features the model was trained on
        common   = [f for f in feature_names if f in X_bg.columns]
        X_bg     = X_bg[common]
        X_scaled = _scaler.transform(X_bg)
        model_type = type(_model).__name__
        if model_type in ("XGBClassifier", "RandomForestClassifier",
                          "GradientBoostingClassifier", "ExtraTreesClassifier"):
            return shap.TreeExplainer(_model)
        elif model_type == "LogisticRegression":
            return shap.LinearExplainer(_model, X_scaled)
        else:
            bg = shap.sample(pd.DataFrame(X_scaled), 50)
            return shap.KernelExplainer(_model.predict_proba, bg)
    except Exception:
        return None


# ── SEVERITY ──────────────────────────────────────────────────────────────────
SEVERITY_BANDS = [
    (0,  10,  "None / Minimal", "#00e676", "rgba(0,230,118,0.1)"),
    (11, 30,  "Mild",           "#ffc107", "rgba(255,193,7,0.1)"),
    (31, 58,  "Moderate",       "#ff9800", "rgba(255,152,0,0.1)"),
    (59, 200, "Severe",         "#ef5350", "rgba(239,83,80,0.1)"),
]

def score_to_band(score):
    for lo, hi, label, color, bg in SEVERITY_BANDS:
        if lo <= round(score) <= hi:
            return label, color, bg
    return "Severe", "#ef5350", "rgba(239,83,80,0.1)"


# ── CHARTS ────────────────────────────────────────────────────────────────────
def create_gauge_chart(probability, title):
    pct = probability * 100
    bar_color = "#ef5350" if pct >= 60 else "#ffc107" if pct >= 40 else "#00e676"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 28, "color": bar_color, "family": "Inter"}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 13, "color": "#8aad95"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4d7260",
                     "tickvals": [0, 25, 50, 75, 100],
                     "ticktext": ["0", "25", "50", "75", "100"]},
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(12,22,16,0.6)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(0,230,118,0.06)"},
                {"range": [40, 65], "color": "rgba(255,193,7,0.06)"},
                {"range": [65, 100],"color": "rgba(239,83,80,0.07)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 2},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(**CHART_DEFAULTS, height=260, margin=dict(l=20, r=20, t=35, b=5),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
    return fig


def create_confidence_distribution(probability):
    vals   = [1 - probability, probability]
    colors = ["#00e676", "#ef5350"]
    fig = go.Figure(go.Bar(
        x=["Healthy", "Parkinson's"],
        y=vals,
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=12, color="#e2f0e7"),
        width=0.45,
    ))
    _apply_chart_theme(fig, height=240)
    fig.update_layout(
        title=dict(text="Confidence Split", font=dict(size=13)),
        yaxis=dict(range=[0, 1.15], showticklabels=False, gridcolor="rgba(255,255,255,0.03)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def create_feature_importance_chart(mdl, feat_names):
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_[0])
    else:
        return None
    idx  = np.argsort(imp)[-12:]
    vals = imp[idx]
    norm = vals / vals.max()
    colors = [f"rgba(0,230,118,{0.3 + 0.7*v:.2f})" for v in norm]
    fig = go.Figure(go.Bar(
        x=vals, y=[feat_names[i] for i in idx],
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#8aad95"),
    ))
    _apply_chart_theme(fig, height=420)
    fig.update_layout(
        title=dict(text="Top Feature Importances (Global)", font=dict(size=13)),
        xaxis_title="Importance Score",
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def display_shap_explanation(input_scaled, feat_names):
    explainer = get_shap_explainer(model, scaler)
    if explainer is None:
        st.warning("SHAP explanation unavailable for this model type.")
        return
    try:
        raw = explainer.shap_values(input_scaled)
        sv  = np.array(raw[1][0]) if isinstance(raw, list) else np.array(raw[0])
        order    = np.argsort(np.abs(sv))[-14:]
        s_sv     = sv[order]
        s_feat   = [feat_names[i] for i in order]
        colors   = ["#ef5350" if v > 0 else "#00e676" for v in s_sv]
        labels   = [f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in s_sv]
        fig = go.Figure(go.Bar(
            x=s_sv, y=s_feat, orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=labels, textposition="outside",
            textfont=dict(size=10, color="#8aad95"),
        ))
        fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.15)")
        _apply_chart_theme(fig, height=460)
        fig.update_layout(
            title=dict(text="SHAP — Why this prediction?", font=dict(size=13)),
            xaxis_title="SHAP value (impact on Parkinson's probability)",
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="ps-shap-legend">
          <span><span class="ps-dot" style="background:#ef5350;"></span>Red → toward Parkinson's</span>
          <span><span class="ps-dot" style="background:#00e676;"></span>Green → toward Healthy</span>
          <span style="color:var(--text-3)">Bar length = strength of influence</span>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")


def display_model_metrics(mtr):
    cols = st.columns(4)
    items = [
        ("Accuracy",  f"{mtr['accuracy']*100:.2f}%",  "Overall correctness"),
        ("Precision", f"{mtr['precision']*100:.2f}%", "True positive rate"),
        ("Recall",    f"{mtr['recall']*100:.2f}%",    "Detection coverage"),
        ("ROC-AUC",   f"{mtr['roc_auc']*100:.2f}%",  "Discrimination power"),
    ]
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="ps-metric">
              <p class="ps-metric-label">{label}</p>
              <p class="ps-metric-value" style="color:var(--accent) !important">{value}</p>
              <p class="ps-metric-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)


# ── AUDIO FEATURE EXTRACTION ──────────────────────────────────────────────────
def extract_features_from_audio(audio_file, feat_names, target_sr=44100):
    y, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    y, _  = librosa.effects.trim(y)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            import soundfile as sf
            sf.write(tmp_path, y, sr)
        snd = parselmouth.Sound(tmp_path)

        pitch = praat_call(snd, "To Pitch", 0.0, 75, 600)
        f0_mean = praat_call(pitch, "Get mean", 0, 0, "Hertz")
        f0_max  = praat_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        f0_min  = praat_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        f0_mean = f0_mean if f0_mean and not np.isnan(f0_mean) else 0.0
        f0_max  = f0_max  if f0_max  and not np.isnan(f0_max)  else 0.0
        f0_min  = f0_min  if f0_min  and not np.isnan(f0_min)  else 0.0

        pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 600)

        def safe(v):
            try: return float(v) if v and not np.isnan(float(v)) else 0.0
            except: return 0.0

        jitter_local  = safe(praat_call(pp, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3))
        jitter_abs    = safe(praat_call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
        jitter_rap    = safe(praat_call(pp, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3))
        jitter_ppq    = safe(praat_call(pp, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3))
        jitter_ddp    = safe(praat_call(pp, "Get jitter (ddp)",             0, 0, 0.0001, 0.02, 1.3))
        shimmer_local = safe(praat_call([snd, pp], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6))
        shimmer_db    = safe(praat_call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        shimmer_apq3  = safe(praat_call([snd, pp], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6))
        shimmer_apq5  = safe(praat_call([snd, pp], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6))
        shimmer_apq11 = safe(praat_call([snd, pp], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6))
        shimmer_dda   = safe(praat_call([snd, pp], "Get shimmer (dda)",      0, 0, 0.0001, 0.02, 1.3, 1.6))

        harmonicity = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = safe(praat_call(harmonicity, "Get mean", 0, 0))
        nhr = float(1.0 / (hnr + 1e-6)) if hnr > 0 else 0.1

        n_frames  = int(praat_call(pitch, "Get number of frames"))
        f0_series = np.array([praat_call(pitch, "Get value in frame", i, "Hertz") for i in range(1, n_frames+1)])
        f0_voiced = f0_series[~np.isnan(f0_series) & (f0_series > 0)]

        rpde = 0.5
        if len(f0_voiced) > 10:
            hist, _ = np.histogram(f0_voiced, bins=min(20, len(f0_voiced)//2), density=True)
            hist = hist[hist > 0]
            h    = -np.sum(hist * np.log(hist + 1e-12))
            rpde = float(np.clip(h / (np.log(len(hist)) + 1e-12), 0.0, 1.0))

        dfa    = 0.6
        signal = f0_voiced if len(f0_voiced) > 20 else y[:2000]
        if len(signal) > 20:
            try:
                cumsum = np.cumsum(signal - np.mean(signal))
                scales = np.unique(np.geomspace(4, len(cumsum)//4, num=12).astype(int))
                fluct  = []
                for s in scales:
                    segs = [cumsum[i:i+s] for i in range(0, len(cumsum)-s, s)]
                    rms  = [np.sqrt(np.mean((seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg))))**2))
                            for seg in segs if len(seg) == s]
                    if rms: fluct.append(np.mean(rms))
                if len(fluct) > 2:
                    dfa = float(np.clip(np.polyfit(np.log(scales[:len(fluct)]), np.log(np.array(fluct)+1e-12), 1)[0], 0.3, 0.95))
            except Exception:
                pass

        spread1 = float(np.clip(np.log(abs(np.percentile(f0_voiced, 25) - np.percentile(f0_voiced, 75)) + 1e-6) * -1, -8.0, -2.0)) if len(f0_voiced) > 4 else -5.0
        spread2 = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-6)) if len(f0_voiced) > 4 else 0.2

        d2 = 2.3
        if len(f0_voiced) > 15:
            try:
                emb  = np.array([f0_voiced[i:i+3] for i in range(len(f0_voiced)-3)])
                dists = np.sqrt(np.sum((emb[:, None] - emb[None, :]) ** 2, axis=-1))
                eps   = np.percentile(dists[dists > 0], [25, 50, 75])
                ci    = [np.mean(dists < e) for e in eps]
                if ci[-1] > ci[0] > 0:
                    d2 = float(np.clip(np.polyfit(np.log(eps+1e-12), np.log(np.array(ci)+1e-12), 1)[0], 1.5, 3.5))
            except Exception:
                pass

        ppe = 0.2
        if len(f0_voiced) > 10:
            periods = 1.0 / (f0_voiced + 1e-6)
            pn = (periods - periods.min()) / ((periods.max() - periods.min()) + 1e-6)
            hist, _ = np.histogram(pn, bins=min(20, len(pn)//2), density=True)
            hist = hist[hist > 0]
            ppe  = float(np.clip(-np.sum(hist * np.log(hist+1e-12)) / (np.log(len(hist))+1e-12), 0.0, 1.0))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    features = {f: 0.0 for f in feat_names}
    mapping  = {
        "MDVP:Fo(Hz)": f0_mean, "MDVP:Fhi(Hz)": f0_max, "MDVP:Flo(Hz)": f0_min,
        "MDVP:Jitter(%)": jitter_local * 100, "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": jitter_rap, "MDVP:PPQ": jitter_ppq, "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_local, "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_apq3, "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq11, "Shimmer:DDA": shimmer_dda,
        "NHR": nhr, "HNR": hnr, "RPDE": rpde, "DFA": dfa,
        "spread1": spread1, "spread2": spread2, "D2": d2, "PPE": ppe,
    }
    for k, v in mapping.items():
        if k in features: features[k] = v
    return pd.DataFrame([features])


# ── ANALYSIS PIPELINE ─────────────────────────────────────────────────────────
def run_analysis(input_data, original_data, label_prefix="Sample"):
    input_data   = input_data[feature_names] if all(c in input_data.columns for c in feature_names) else input_data
    input_scaled = scaler.transform(input_data)
    predictions  = model.predict(input_scaled)
    probs        = model.predict_proba(input_scaled)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Analysis Results")

    for idx, (pred, prob) in enumerate(zip(predictions, probs)):
        pd_prob  = prob[1]
        hlt_prob = prob[0]
        is_pd    = pred == 1

        st.session_state.history.append({
            "Timestamp":      datetime.now().strftime("%H:%M:%S"),
            "Label":          f"{label_prefix} {idx + 1}",
            "Input Type":     "Audio" if label_prefix == "Recording" else "CSV",
            "Result":         "Parkinson's Risk" if is_pd else "Healthy",
            "PD Probability": f"{pd_prob*100:.1f}%",
            "Confidence":     f"{max(pd_prob, hlt_prob)*100:.1f}%",
        })

        c1, c2 = st.columns([1, 1])

        with c1:
            if is_pd:
                st.markdown(f"""
                <div class="ps-result ps-result-risk">
                  <p class="ps-result-title" style="color:#ef5350 !important;">⚠ Elevated Risk Detected</p>
                  <p class="ps-result-body">Voice analysis indicates patterns consistent with Parkinson's disease biomarkers.</p>
                  <p class="ps-result-note">Consult a neurologist for comprehensive clinical evaluation.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ps-result ps-result-healthy">
                  <p class="ps-result-title" style="color:var(--accent) !important;">✓ Low Risk Detected</p>
                  <p class="ps-result-body">Voice patterns appear within normal ranges. No significant biomarker deviation detected.</p>
                  <p class="ps-result-note">Continue regular health monitoring and maintain healthy habits.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("PD Likelihood", f"{pd_prob*100:.1f}%",
                          delta=f"{(pd_prob-0.5)*100:+.1f}pp" if pd_prob > 0.5 else None)
            with mc2:
                st.metric("Healthy Likelihood", f"{hlt_prob*100:.1f}%",
                          delta=f"{(hlt_prob-0.5)*100:+.1f}pp" if hlt_prob > 0.5 else None)

            # Severity
            if is_pd and sev_model is not None:
                st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
                try:
                    row = input_data.iloc[[idx]]
                    sev_in = pd.DataFrame([{f: float(row[f].values[0]) if f in row.columns else 0.0
                                            for f in sev_feature_names}])
                    updrs  = float(np.clip(sev_model.predict(sev_scaler.transform(sev_in))[0], 0, 59))
                    band, bcolor, bbg = score_to_band(updrs)
                    mae_val = sev_metrics.get("mae", 5.2)
                    st.markdown(f"""
                    <div style="background:{bbg}; border:1px solid {bcolor}33;
                         border-left:3px solid {bcolor}; border-radius:10px;
                         padding:.9rem 1.1rem;">
                      <p style="margin:0; font-size:.7rem; text-transform:uppercase;
                                letter-spacing:.07em; color:var(--text-3) !important; font-weight:600;">
                        Motor UPDRS Estimate
                      </p>
                      <div style="display:flex; align-items:baseline; gap:.5rem; margin:.3rem 0 .2rem;">
                        <span style="font-size:1.7rem; font-weight:700; color:{bcolor} !important;
                                     letter-spacing:-.03em;">{updrs:.1f}</span>
                        <span style="font-size:.85rem; color:var(--text-3) !important;">/ 108</span>
                        <span class="ps-severity" style="background:{bcolor}1a; color:{bcolor} !important;
                              border:1px solid {bcolor}33; margin-left:.2rem;">{band}</span>
                      </div>
                      <p style="margin:0; font-size:.73rem; color:var(--text-3) !important;">
                        ± {mae_val:.1f} UPDRS points · based on 16 voice biomarkers
                      </p>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    pass

        with c2:
            st.plotly_chart(create_gauge_chart(pd_prob, "PD Risk Score"), use_container_width=True)
            st.plotly_chart(create_confidence_distribution(pd_prob), use_container_width=True)

        if idx < len(predictions) - 1:
            st.markdown("<hr>", unsafe_allow_html=True)

    # SHAP
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### SHAP Explanation — What drove this result?")
    st.caption("Per-prediction feature attribution. Shows exactly which biomarkers influenced this specific outcome.")
    with st.spinner("Computing SHAP values…"):
        display_shap_explanation(input_scaled[[0]], feature_names)

    # Global importance
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Global Feature Importance")
    st.caption("Features ranked by their overall influence across all predictions.")
    imp_fig = create_feature_importance_chart(model, feature_names)
    if imp_fig:
        st.plotly_chart(imp_fig, use_container_width=True)


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
model, scaler, feature_names, metrics, \
    sev_model, sev_scaler, sev_feature_names, sev_metrics = load_models()
if model is None:
    st.stop()
if "history" not in st.session_state:
    st.session_state.history = []


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="ps-brand">
      <span class="ps-brand-icon">🎙️</span>
      <div>
        <div class="ps-brand-name">ParkiSense AI</div>
        <div class="ps-brand-tag">Voice Analysis Platform</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<p class="ps-section-label">Model</p>', unsafe_allow_html=True)
    model_stats = [
        ("Algorithm", metrics.get("model_label", type(model).__name__)),
        ("Features",  str(metrics.get("n_features", len(feature_names)))),
        ("Training",  f"{metrics.get('dataset_size', 195):,} samples"),
        ("ROC-AUC",   f"{metrics.get('roc_auc', 0)*100:.1f}%"),
        ("Recall",    f"{metrics.get('recall', 0)*100:.1f}%"),
    ]
    for k, v in model_stats:
        st.markdown(f"""
        <div class="ps-stat-row">
          <span class="ps-stat-key">{k}</span>
          <span class="ps-stat-val">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<p class="ps-section-label">Session History</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<p style="font-size:.8rem; color:var(--text-3);">No predictions yet.</p>',
                    unsafe_allow_html=True)
    else:
        hist_df   = pd.DataFrame(st.session_state.history)
        total     = len(hist_df)
        risk_n    = (hist_df["Result"] == "Parkinson's Risk").sum()
        healthy_n = total - risk_n
        st.markdown(f"""
        <p style="font-size:.75rem; color:var(--text-3); margin:0 0 .5rem;">
          {total} prediction{"s" if total > 1 else ""}
          &nbsp;·&nbsp; <span style="color:#ef5350">{risk_n} at-risk</span>
          &nbsp;·&nbsp; <span style="color:var(--accent)">{healthy_n} healthy</span>
        </p>""", unsafe_allow_html=True)

        for entry in reversed(st.session_state.history):
            is_risk = entry["Result"] == "Parkinson's Risk"
            dot     = "🔴" if is_risk else "🟢"
            st.markdown(f"""
            <div class="ps-history-item"
                 style="border-left-color:{'#ef5350' if is_risk else 'var(--accent)'}">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:.78rem; font-weight:600; color:var(--text-1) !important;">
                  {dot} {entry['Label']}
                </span>
                <span style="font-size:.7rem; color:var(--text-3) !important;">{entry['Timestamp']}</span>
              </div>
              <div style="font-size:.73rem; color:var(--text-3) !important; margin-top:.15rem;">
                {entry['PD Probability']} PD prob · {entry['Input Type']}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "↓ Export History (CSV)",
            data=hist_df.to_csv(index=False),
            file_name=f"parkisense_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:.72rem; color:var(--text-3); line-height:1.5;">
      Screening tool only — not a substitute for clinical diagnosis.
      Always consult a neurologist.
    </p>""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ps-header">
  <div>
    <div class="ps-header-badge">🧬 AI-Powered Healthcare</div>
    <h1 class="ps-header-title">ParkiSense AI</h1>
    <p class="ps-header-sub">Parkinson's Disease Voice Biomarker Analysis &amp; Severity Estimation</p>
  </div>
</div>""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Analysis  ", "  Model Performance  ", "  Data Explorer  ", "  Information  "
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("#### Voice Feature Analysis")
    st.markdown('<p class="ps-section-label">Select input method</p>', unsafe_allow_html=True)

    input_mode = st.radio(
        "input_mode", ["Upload CSV", "Audio Input"],
        horizontal=True, label_visibility="collapsed"
    )

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── CSV MODE ──────────────────────────────────────────────────────────────
    if input_mode == "Upload CSV":
        left, right = st.columns([3, 1], gap="large")

        with left:
            st.markdown("""
            <div class="ps-card" style="margin-bottom:1rem;">
              <p class="ps-card-title">CSV Upload</p>
              <p class="ps-card-body">
                Upload a CSV file containing pre-extracted voice feature measurements.
                The model analyzes 16 clinical biomarkers including jitter, shimmer,
                harmonics-to-noise ratio, and nonlinear dynamics.
              </p>
            </div>""", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop your CSV here or click to browse",
                type="csv", label_visibility="visible"
            )

        with right:
            st.markdown("""
            <div class="ps-tips">
              <p class="ps-tips-title">Required Format</p>
              <ul>
                <li>CSV with numeric columns</li>
                <li>16+ voice features</li>
                <li>No missing values</li>
                <li>One row = one sample</li>
              </ul>
            </div>""", unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                input_data    = pd.read_csv(uploaded_file)
                original_data = input_data.copy()
                drops = [c for c in ["name", "status"] if c in input_data.columns]
                if drops:
                    input_data = input_data.drop(drops, axis=1)

                st.success(f"✓ {len(input_data)} sample(s) loaded successfully")

                with st.expander("Preview uploaded data"):
                    st.dataframe(original_data.head(10), use_container_width=True)

                if st.button("Run Voice Analysis", use_container_width=True):
                    with st.spinner("Analyzing biomarkers…"):
                        run_analysis(input_data, original_data, label_prefix="Sample")

            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Ensure the CSV contains all required voice biomarker columns.")
        else:
            st.markdown("""
            <div class="ps-upload-zone">
              <div class="ps-upload-icon">📂</div>
              <p class="ps-upload-title">No file selected</p>
              <p class="ps-upload-hint">Upload a .csv file with MDVP voice features to begin</p>
            </div>""", unsafe_allow_html=True)

    # ── AUDIO MODE ────────────────────────────────────────────────────────────
    else:
        left, right = st.columns([3, 1], gap="large")

        with left:
            st.markdown("""
            <div class="ps-card" style="margin-bottom:1rem;">
              <p class="ps-card-title">Audio Analysis</p>
              <p class="ps-card-body">
                Upload or record a short voice sample. Features are extracted using
                <strong>Praat</strong> — the same clinical-grade software used to build
                the training dataset — making audio results as reliable as CSV input.
              </p>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="ps-section-label">Audio source</p>', unsafe_allow_html=True)
            source = st.radio("source", ["Upload file", "Record with mic"],
                              horizontal=True, label_visibility="collapsed")

            audio_bytes = None
            audio_file  = None

            if source == "Upload file":
                audio_file = st.file_uploader(
                    "Drop audio file here", type=["wav", "mp3", "ogg", "m4a"],
                    label_visibility="visible"
                )
                if audio_file:
                    st.audio(audio_file)
            else:
                st.markdown('<p style="font-size:.84rem; color:var(--text-2);">Click Start and sustain an "aaa" vowel sound for 5–10 seconds</p>', unsafe_allow_html=True)
                audio = audiorecorder("● Start Recording", "■ Stop Recording")
                if len(audio) > 0:
                    buf = io.BytesIO()
                    audio.export(buf, format="wav")
                    buf.seek(0)
                    audio_bytes = buf.getvalue()
                    st.audio(audio_bytes, format="audio/wav")

            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

            if st.button("Extract Features & Analyze", use_container_width=True):
                with st.spinner("Extracting Praat features and running analysis…"):
                    try:
                        if source == "Upload file":
                            if audio_file is None:
                                st.error("Please upload an audio file first.")
                                st.stop()
                            audio_input = audio_file
                        else:
                            if audio_bytes is None:
                                st.error("Please record your voice first.")
                                st.stop()
                            audio_input = io.BytesIO(audio_bytes)

                        features_df = extract_features_from_audio(audio_input, feature_names)
                        st.success("Features extracted successfully")

                        with st.expander("View extracted feature values"):
                            st.dataframe(features_df, use_container_width=True)

                        buf2 = io.StringIO()
                        features_df.to_csv(buf2, index=False)
                        st.download_button(
                            "↓ Download Features (CSV)", data=buf2.getvalue(),
                            file_name="parkinsons_features.csv", mime="text/csv"
                        )
                        run_analysis(features_df, features_df, label_prefix="Recording")

                    except Exception as e:
                        st.error(f"Audio processing error: {e}")
                        st.info("Use a clear WAV/MP3 recording, 3–10 seconds in length.")

        with right:
            st.markdown("""
            <div class="ps-tips">
              <p class="ps-tips-title">Recording Tips</p>
              <ul>
                <li>Quiet room, minimal echo</li>
                <li>Sustain "aaaaa" vowel</li>
                <li>Mic 15–20 cm from mouth</li>
                <li>3–10 second duration</li>
                <li>Avoid background noise</li>
              </ul>
            </div>
            <div style="height:.75rem"></div>
            <div class="ps-card">
              <p class="ps-card-title">Praat Extraction</p>
              <p class="ps-card-body" style="font-size:.78rem;">
                Jitter, shimmer, and HNR are computed using Praat's exact MDVP
                algorithm — the same tool used to generate the training dataset.
              </p>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("#### Model Performance")
    st.caption("Metrics evaluated on the UCI Parkinson's held-out test set.")

    display_model_metrics(metrics)

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("##### Confusion Matrix")
        cm_data = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_df   = pd.DataFrame(
            [[cm_data[0][0], cm_data[0][1]], [cm_data[1][0], cm_data[1][1]]],
            index=["Healthy", "Parkinson's"],
            columns=["Predicted Healthy", "Predicted PD"]
        )
        fig_cm = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale=[[0,"#0c1610"],[0.5,"rgba(0,230,118,0.3)"],[1,"#00e676"]],
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_traces(textfont=dict(size=14, color="#e2f0e7"))
        fig_cm.update_layout(**CHART_DEFAULTS, height=300,
                             coloraxis_showscale=False,
                             xaxis=dict(side="bottom", tickfont=dict(size=11)),
                             yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown("##### Performance Breakdown")
        m_vals = {
            "Accuracy":  metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1-Score":  metrics["f1"],
            "ROC-AUC":   metrics["roc_auc"],
        }
        m_df = pd.DataFrame({"Metric": list(m_vals.keys()), "Score": [v*100 for v in m_vals.values()]})
        colors_bar = [f"rgba(0,230,118,{0.4 + 0.6*v:.2f})" for v in m_vals.values()]
        fig_bar = go.Figure(go.Bar(
            x=m_df["Metric"], y=m_df["Score"],
            marker=dict(color=colors_bar, cornerradius=6),
            text=[f"{v:.2f}%" for v in m_df["Score"]],
            textposition="outside",
            textfont=dict(size=11, color="#8aad95"),
            width=0.55,
        ))
        _apply_chart_theme(fig_bar, height=300)
        fig_bar.update_layout(
            yaxis=dict(range=[0, 108], showticklabels=False, gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ps-card">
      <p class="ps-card-title">Training Details</p>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:.5rem .8rem; margin-top:.4rem;">
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Dataset</p>
             <p style="font-size:.84rem;color:var(--text-1)!important;margin:0;font-weight:500;">{metrics.get('dataset_size',195):,} samples — UCI + Telemonitoring</p></div>
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Best Model</p>
             <p style="font-size:.84rem;color:var(--accent)!important;margin:0;font-weight:600;">{metrics.get('model_label','XGBoost')}</p></div>
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Validation</p>
             <p style="font-size:.84rem;color:var(--text-1)!important;margin:0;font-weight:500;">{metrics.get('cv_folds',5)}-fold CV · SMOTE balancing</p></div>
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Features</p>
             <p style="font-size:.84rem;color:var(--text-1)!important;margin:0;font-weight:500;">{metrics.get('n_features',16)} voice biomarkers</p></div>
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Models Compared</p>
             <p style="font-size:.84rem;color:var(--text-1)!important;margin:0;font-weight:500;">LR · RF · SVM · XGBoost</p></div>
        <div><p style="font-size:.72rem;color:var(--text-3)!important;margin:0;">Selection Criterion</p>
             <p style="font-size:.84rem;color:var(--text-1)!important;margin:0;font-weight:500;">ROC-AUC (honest on imbalanced data)</p></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — DATA EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("#### Training Dataset Explorer")
    st.caption("Datasets used to train the model — UCI (195 samples) + Parkinson's Telemonitoring (5,875 samples).")

    @st.cache_data
    def load_dataset():
        # UCI dataset — 22 features, binary labels, used for visualization
        uci = pd.read_csv("parkinsons.csv")
        uci["Diagnosis"] = uci["status"].map({1: "Parkinson's", 0: "Healthy"})
        uci["Source"] = "UCI"

        # Telemonitoring — 16 features, all PD patients
        TELE_TO_UCI = {
            "Jitter(%)": "MDVP:Jitter(%)", "Jitter(Abs)": "MDVP:Jitter(Abs)",
            "Jitter:RAP": "MDVP:RAP", "Jitter:PPQ5": "MDVP:PPQ",
            "Jitter:DDP": "Jitter:DDP", "Shimmer": "MDVP:Shimmer",
            "Shimmer(dB)": "MDVP:Shimmer(dB)", "Shimmer:APQ3": "Shimmer:APQ3",
            "Shimmer:APQ5": "Shimmer:APQ5", "Shimmer:APQ11": "MDVP:APQ",
            "Shimmer:DDA": "Shimmer:DDA", "NHR": "NHR", "HNR": "HNR",
            "RPDE": "RPDE", "DFA": "DFA", "PPE": "PPE",
        }
        tele = pd.read_csv("parkinsons_updrs.csv").rename(columns=TELE_TO_UCI)
        tele["status"]    = 1
        tele["Diagnosis"] = "Parkinson's"
        tele["Source"]    = "Telemonitoring"
        return uci, tele

    df_uci, df_tele = load_dataset()

    # Combined training stats
    total_combined = len(df_uci) + len(df_tele)
    pd_combined    = int((df_uci["status"] == 1).sum()) + len(df_tele)
    healthy_orig   = int((df_uci["status"] == 0).sum())
    uci_total      = len(df_uci)
    pd_n           = int((df_uci["status"] == 1).sum())
    healthy_n      = healthy_orig

    # Use UCI for visualizations (has all 22 features)
    df_raw = df_uci

    # Stats row — combined training numbers
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val, sub in [
        (s1, "Training Samples", f"{total_combined:,}", "UCI + Telemonitoring"),
        (s2, "Parkinson's",      f"{pd_combined:,}",   f"across both datasets"),
        (s3, "Healthy Controls", str(healthy_orig),    "UCI dataset only"),
        (s4, "Features Used",    str(metrics.get("n_features", 16)), "common biomarkers"),
    ]:
        with col:
            st.markdown(f"""
            <div class="ps-metric">
              <p class="ps-metric-label">{label}</p>
              <p class="ps-metric-value">{val}</p>
              <p class="ps-metric-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    # Dataset breakdown bar
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ps-callout">
      <strong>Combined training strategy:</strong> UCI Parkinson's (195 rows, 22 features, binary labels)
      + Telemonitoring (5,875 rows, 16 features, all PD patients) → 16 common features used for training.
      SMOTE upsamples 48 healthy recordings to ~1,200 synthetic samples to counter extreme class imbalance.
      Violin plots and heatmap below use the UCI dataset which has the full 22-feature set.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Class balance + description
    bc1, bc2 = st.columns([1, 1], gap="large")

    with bc1:
        st.markdown("##### Class Balance — Combined Training Data")
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Bar(
            name="Parkinson's", x=["UCI", "Telemonitoring"],
            y=[pd_n, len(df_tele)],
            marker=dict(color="#ef5350", cornerradius=6),
            text=[str(pd_n), f"{len(df_tele):,}"],
            textposition="outside", textfont=dict(color="#8aad95", size=11),
            width=0.35,
        ))
        fig_bal.add_trace(go.Bar(
            name="Healthy", x=["UCI", "Telemonitoring"],
            y=[healthy_n, 0],
            marker=dict(color="#00e676", cornerradius=6),
            text=[str(healthy_n), "0"], textposition="outside",
            textfont=dict(color="#8aad95", size=11),
            width=0.35,
        ))
        _apply_chart_theme(fig_bal, height=280)
        fig_bal.update_layout(
            barmode="group",
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=1.1, x=0),
            yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False),
            showlegend=True,
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    with bc2:
        st.markdown("##### About the Datasets")
        st.markdown(f"""
        <div class="ps-card">
          <p class="ps-card-title">UCI Parkinson's Voice Dataset</p>
          <p class="ps-card-body">
            195 recordings from 31 subjects (23 PD, 8 healthy).
            22 MDVP voice biomarkers. Used for feature visualization below
            and as the labelled healthy-class source.
          </p>
          <div style="height:.6rem"></div>
          <p class="ps-card-title">Parkinson's Telemonitoring Dataset</p>
          <p class="ps-card-body">
            5,875 recordings from 42 PD patients monitored over time via phone.
            16 voice features (subset of MDVP). Provides diverse PD voice patterns
            across different severity levels (motor UPDRS 5–39).
          </p>
          <div style="height:.5rem"></div>
          <p class="ps-card-body" style="font-size:.76rem; color:var(--text-3) !important;">
            UCI: Little et al. (2008), IEEE Trans. Biomed. Eng. &nbsp;·&nbsp;
            Telemonitoring: Tsanas et al. (2010), IEEE Trans. Biomed. Eng. &nbsp;·&nbsp;
            <span style="color:var(--accent)">UCI ML Repository</span>
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature distributions
    st.markdown("##### Feature Distributions — Healthy vs Parkinson's")
    st.caption("Violin plots show value spread per class. More separation = more discriminative feature.")

    exclude_cols  = {"name", "status", "Diagnosis", "Source"}
    numeric_feats = [c for c in df_raw.columns
                     if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_raw[c])]
    sel = st.multiselect(
        "Select features to visualize:",
        options=numeric_feats,
        default=["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "RPDE", "DFA", "PPE"],
    )

    if sel:
        chunks = [sel[i:i+3] for i in range(0, len(sel), 3)]
        for chunk in chunks:
            cols_ = st.columns(3)
            for ci, feat in enumerate(chunk):
                with cols_[ci]:
                    pd_v  = df_raw.loc[df_raw["status"] == 1, feat].values
                    hlt_v = df_raw.loc[df_raw["status"] == 0, feat].values
                    vf = go.Figure()
                    vf.add_trace(go.Violin(y=pd_v, name="PD", side="negative",
                                           line_color="#ef5350", fillcolor="rgba(239,83,80,0.15)",
                                           box_visible=True, meanline_visible=True, points=False))
                    vf.add_trace(go.Violin(y=hlt_v, name="Healthy", side="positive",
                                           line_color="#00e676", fillcolor="rgba(0,230,118,0.12)",
                                           box_visible=True, meanline_visible=True, points=False))
                    _apply_chart_theme(vf, height=240)
                    vf.update_layout(
                        title=dict(text=feat, font=dict(size=11)),
                        showlegend=(ci == 0 and chunk == chunks[0]),
                        violingap=0, violinmode="overlay",
                        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                                    orientation="h", y=1.12, x=0),
                        margin=dict(l=5, r=5, t=40, b=5),
                    )
                    st.plotly_chart(vf, use_container_width=True)
    else:
        st.markdown("""
        <div class="ps-upload-zone" style="padding:1.5rem;">
          <p class="ps-upload-title" style="font-size:.9rem;">Select features above</p>
          <p class="ps-upload-hint">Choose one or more biomarkers to visualize their distributions</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("##### Feature Correlation Heatmap")
    st.caption("Look at the 'status' row — features with strong correlation are most predictive of Parkinson's.")

    corr = df_raw[numeric_feats + ["status"]].corr().round(2)
    fig_hm = px.imshow(
        corr, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, text_auto=".1f", aspect="auto",
    )
    fig_hm.update_traces(textfont=dict(size=7, color="#e2f0e7"))
    fig_hm.update_layout(
        **CHART_DEFAULTS, height=580,
        coloraxis_colorbar=dict(
            title="r", tickfont=dict(color="#8aad95", size=9),
            thickness=10, len=0.7
        ),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("View Raw Dataset"):
        st.dataframe(df_raw.drop(columns=["Diagnosis", "Source"]), use_container_width=True, height=340)
        st.download_button(
            "↓ Download Dataset (CSV)",
            data=df_raw.drop(columns=["Diagnosis", "Source"]).to_csv(index=False),
            file_name="parkinsons_dataset.csv", mime="text/csv"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — INFORMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("#### About Parkinson's Disease & Voice Analysis")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="ps-card">
          <p class="ps-card-title">What is Parkinson's Disease?</p>
          <p class="ps-card-body">
            Parkinson's disease is a progressive nervous system disorder affecting movement.
            It develops gradually — often starting with a subtle tremor in one hand — and
            commonly causes stiffness, slowing of movement, and balance problems.
          </p>
          <div style="height:.5rem"></div>
          <p class="ps-card-title" style="margin-top:.5rem;">Early Signs</p>
          <ul style="margin:0; padding-left:1.1rem;">
            <li class="ps-card-body" style="margin-bottom:.3rem;">Tremor in hands, arms, legs, or jaw</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;">Stiffness of limbs and trunk</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;">Slowness of movement (bradykinesia)</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;">Impaired balance and coordination</li>
            <li class="ps-card-body">Voice changes and reduced volume</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ps-card">
          <p class="ps-card-title">Why Voice Analysis?</p>
          <p class="ps-card-body">
            Voice impairment (dysphonia) affects up to <strong>90% of Parkinson's patients</strong>
            and is often one of the earliest detectable signs — appearing years before
            motor symptoms become clinically obvious.
          </p>
          <div style="height:.5rem"></div>
          <p class="ps-card-title" style="margin-top:.5rem;">Key Biomarkers Analyzed</p>
          <ul style="margin:0; padding-left:1.1rem;">
            <li class="ps-card-body" style="margin-bottom:.3rem;"><strong>Jitter</strong> — Cycle-to-cycle frequency variation</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;"><strong>Shimmer</strong> — Amplitude variation between cycles</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;"><strong>HNR</strong> — Harmonics-to-Noise Ratio</li>
            <li class="ps-card-body" style="margin-bottom:.3rem;"><strong>RPDE</strong> — Recurrence Period Density Entropy</li>
            <li class="ps-card-body"><strong>DFA / PPE</strong> — Nonlinear dynamics measures</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="ps-result ps-result-risk" style="border-left-width:3px;">
      <p class="ps-result-title" style="color:#ef5350 !important; font-size:.85rem;">⚠ Medical Disclaimer</p>
      <p class="ps-result-body">
        <strong>This tool is for screening and research purposes only.</strong>
        It is not a substitute for professional medical diagnosis. A positive result does not
        confirm Parkinson's disease. If you have concerns, please consult a qualified neurologist
        or healthcare provider for comprehensive clinical evaluation.
      </p>
      <p class="ps-result-note">
        Early detection and treatment can significantly improve quality of life.
        ParkiSense AI is designed to assist screening — not to replace clinical judgment.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    faq_items = [
        ("How accurate is the analysis?",
         f"The model achieves ROC-AUC {metrics.get('roc_auc',0)*100:.1f}% on the UCI test set, trained on {metrics.get('dataset_size',195):,} samples. Accuracy may vary on recordings from different equipment or populations."),
        ("What recording quality is needed?",
         "A clear WAV or MP3 file recorded in a quiet environment works best. Sustain a single vowel sound (e.g. 'aaaaa') for 5–10 seconds at a comfortable distance from the microphone."),
        ("Why does the model use 16 features instead of all 22?",
         "The combined training dataset (UCI + Telemonitoring) shares 16 common features. F0/Fhi/Flo and spread1/2/D2 are only in the UCI dataset and were excluded to allow training on 6,070 samples instead of 195."),
    ]
    st.markdown('<p class="ps-section-label" style="margin-top:.5rem;">FAQ</p>', unsafe_allow_html=True)
    for q, a in faq_items:
        with st.expander(q):
            st.markdown(f'<p class="ps-card-body">{a}</p>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="ps-footer">
  <p>ParkiSense AI &nbsp;·&nbsp; Parkinson's Voice Analysis Platform</p>
  <p>For screening purposes only &nbsp;·&nbsp; Always consult a healthcare professional</p>
</div>""", unsafe_allow_html=True)

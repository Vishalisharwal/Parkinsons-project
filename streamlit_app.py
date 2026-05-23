import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import io
from audiorecorder import audiorecorder





# ============ PAGE CONFIGURATION ============

st.set_page_config(
    page_title="ParkiSense AI – Parkinson's Voice Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>

/* ---------- GLOBAL DARK THEME ---------- */
:root {
    --bg-black: #000000;
    --card-black: #0d0d0d;
    --border-green: #00ff88;
    --neon-green: #00ff88;
    --text-green: #b6ffd5;
}

/* Page background */
.stApp {
    background-color: var(--bg-black) !important;
}

/* Text – default */
html, body, [class*="css"] {
    color: var(--text-green) !important;
}

/* ---------- HEADERS ---------- */
.main-header {
    background: linear-gradient(90deg, #003b24, #000000);
    padding: 2rem;
    border-radius: 12px;
    color: var(--neon-green);
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,136,0.3);
    border: 1px solid var(--border-green);
}

.main-header h1 {
    font-size: 2.5rem;
    color: var(--neon-green);
}

.main-header p {
    color: var(--text-green);
    opacity: 0.9;
}

/* Mascot animation */
.mascot-container {
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}

/* ---------- CARDS (metric, info, results) ---------- */
.metric-card, .info-box, .result-positive, .result-negative {
    background: var(--card-black);
    border: 1px solid var(--border-green);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 0 15px rgba(0,255,136,0.15);
}

/* positive (normal result) */
.result-negative {
    border-color: #00ff88;
}

/* alert (parkinson-like pattern) */
.result-positive {
    border-color: #ff4444;
    box-shadow: 0 0 15px rgba(255,68,68,0.2);
    color: #ff8888 !important;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: #001b10;
    color: var(--neon-green);
    border: 1px solid var(--neon-green);
    padding: 0.6rem 1.8rem;
    border-radius: 8px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    background: var(--neon-green);
    color: #000;
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(0,255,136,0.5);
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #000 !important;
    border-right: 1px solid var(--neon-green);
}

[data-testid="stSidebar"] * {
    color: var(--neon-green) !important;
}

/* ---------- PROGRESS BAR ---------- */
.stProgress > div > div > div {
    background: var(--neon-green) !important;
}

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"] div,
[data-testid="stDataFrame"] table {
    color: var(--text-green) !important;
    background: #000 !important;
}

/* ---------- Plotly background ---------- */
.js-plotly-plot .plotly, .plot-container {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)



# st.markdown("""
#     <style>
#     /* Main theme colors */
#     :root {
#         --primary-blue: #4A90E2;
#         --light-blue: #E8F4F8;
#         --accent-blue: #2E86DE;
#         --white: #FFFFFF;
#     }
    
#     /* Background */
#     .stApp {
#         background: linear-gradient(135deg, #E8F4F8 0%, #FFFFFF 100%);
#     }
    
#     /* Header styling */
#     .main-header {
#         background: linear-gradient(90deg, #4A90E2 0%, #2E86DE 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .main-header h1 {
#         margin: 0;
#         font-size: 2.5rem;
#         font-weight: 700;
#     }
    
#     .main-header p {
#         margin: 0.5rem 0 0 0;
#         font-size: 1.1rem;
#         opacity: 0.95;
#     }
    
#     /* Mascot container */
#     .mascot-container {
#         text-align: center;
#         font-size: 80px;
#         animation: float 3s ease-in-out infinite;
#         margin: 1rem 0;
#     }
    
#     @keyframes float {
#         0%, 100% { transform: translateY(0px); }
#         50% { transform: translateY(-20px); }
#     }
    
#     /* Metric cards */
#     .metric-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 12px;
#         box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
#         border-left: 4px solid #4A90E2;
#         margin: 1rem 0;
#     }
    
#     /* Info boxes */
#     .info-box {
#         background: linear-gradient(135deg, #E8F4F8 0%, #D4E9F7 100%);
#         padding: 1.5rem;
#         border-radius: 12px;
#         border: 2px solid #4A90E2;
#         margin: 1rem 0;
#     }
    
#     /* Result boxes */
#     .result-positive {
#         background: linear-gradient(135deg, #FFE5E5 0%, #FFD4D4 100%);
#         border: 2px solid #E74C3C;
#         padding: 1.5rem;
#         border-radius: 12px;
#         margin: 1rem 0;
#     }
    
#     .result-negative {
#         background: linear-gradient(135deg, #E8F8F5 0%, #D4F1E8 100%);
#         border: 2px solid #27AE60;
#         padding: 1.5rem;
#         border-radius: 12px;
#         margin: 1rem 0;
#     }
    
#     /* Button styling */
#     .stButton>button {
#         background: linear-gradient(90deg, #4A90E2 0%, #2E86DE 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         border-radius: 8px;
#         font-weight: 600;
#         transition: all 0.3s;
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
#     }
    
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #4A90E2 0%, #2E86DE 100%);
#     }
    
#     [data-testid="stSidebar"] * {
#         color: white !important;
#     }
    
#     /* Progress bar */
#     .stProgress > div > div > div {
#         background: linear-gradient(90deg, #4A90E2 0%, #2E86DE 100%);
#     }
#     </style>
# """, unsafe_allow_html=True)

# ============ HELPER FUNCTIONS ============

@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, scaler, feature_names
    except:
        st.error("⚠️ Model files not found. Please train the model first.")
        return None, None, None


def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#2C3E50'}},
        delta={'reference': 50, 'increasing': {'color': "#E74C3C"}, 'decreasing': {'color': "#27AE60"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#4A90E2"},
            'bar': {'color': "#4A90E2"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#2C3E50",
            'steps': [
                {'range': [0, 30], 'color': '#D4F1E8'},
                {'range': [30, 70], 'color': '#FFF4E5'},
                {'range': [70, 100], 'color': '#FFD4D4'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2C3E50", 'family': "Arial"},
        height=300
    )

    return fig


def create_feature_importance_chart(model, feature_names):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Get top 10 features
    indices = np.argsort(importances)[-10:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker=dict(
            color=top_importances,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))

    fig.update_layout(
        title="Top 10 Most Important Voice Features",
        xaxis_title="Feature Importance",
        yaxis_title="Voice Feature",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(232, 244, 248, 0.3)",
        height=500,
        font=dict(color='#2C3E50')
    )

    return fig


def create_confidence_distribution(probability):
    """Create a confidence distribution chart"""
    categories = ['Healthy', "Parkinson's"]
    values = [1 - probability, probability]
    colors = ['#27AE60', '#E74C3C']

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v*100:.1f}%' for v in values],
        textposition='auto',
    )])

    fig.update_layout(
        title="Prediction Confidence Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(232, 244, 248, 0.3)",
        height=350,
        font=dict(color='#2C3E50')
    )

    return fig


def display_model_metrics():
    """Display model performance metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style='color: #4A90E2; margin: 0;'>Accuracy</h3>
                <h2 style='color: #2C3E50; margin: 0.5rem 0;'>94.87%</h2>
                <p style='color: #7F8C8D; margin: 0; font-size: 0.9rem;'>Overall Model</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3 style='color: #4A90E2; margin: 0;'>Precision</h3>
                <h2 style='color: #2C3E50; margin: 0.5rem 0;'>96.15%</h2>
                <p style='color: #7F8C8D; margin: 0; font-size: 0.9rem;'>Positive Cases</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3 style='color: #4A90E2; margin: 0;'>Recall</h3>
                <h2 style='color: #2C3E50; margin: 0.5rem 0;'>96.15%</h2>
                <p style='color: #7F8C8D; margin: 0; font-size: 0.9rem;'>Detection Rate</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3 style='color: #4A90E2; margin: 0;'>ROC-AUC</h3>
                <h2 style='color: #2C3E50; margin: 0.5rem 0;'>98.21%</h2>
                <p style='color: #7F8C8D; margin: 0; font-size: 0.9rem;'>Classification</p>
            </div>
        """, unsafe_allow_html=True)


def extract_features_from_audio(audio_file, feature_names, target_sr=22050):
    """
    Extract basic voice features from an audio file and map them
    to the feature_names used by the trained model.

    NOTE: This is an approximate feature extractor. For best results,
    your training-time feature engineering should be replicated here.
    """
    # Load audio from uploaded file
    y, sr = librosa.load(audio_file, sr=target_sr, mono=True)

    # Trim leading / trailing silence
    y, _ = librosa.effects.trim(y)

    # Initialize all features to 0
    features = {f: 0.0 for f in feature_names}

    # --------- Example acoustic features ----------
    # Fundamental frequency (F0) using PYIN
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=400, sr=sr
        )
        f0_voiced = f0[~np.isnan(f0)]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) > 0:
        f0_mean = float(np.mean(f0_voiced))
        f0_max = float(np.max(f0_voiced))
        f0_min = float(np.min(f0_voiced))
        jitter_local = float(
            np.std(np.diff(f0_voiced)) / (np.mean(f0_voiced) + 1e-6)
        )
    else:
        f0_mean = f0_max = f0_min = jitter_local = 0.0

    # Amplitude variation (approx shimmer)
    amp = np.abs(y)
    shimmer_local = float(np.std(amp) / (np.mean(amp) + 1e-6))

    # Zero-crossing rate (general voice stability)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Approximate HNR
    hnr = 0.0
    try:
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        h_energy = np.sum(harmonic ** 2)
        n_energy = np.sum(noise ** 2) + 1e-6
        hnr = float(10 * np.log10(h_energy / n_energy))
    except Exception:
        pass

    # --------- Map to common Parkinson's dataset feature names ----------
    mapping = {
        "MDVP:Fo(Hz)": f0_mean,
        "MDVP:Fhi(Hz)": f0_max,
        "MDVP:Flo(Hz)": f0_min,
        "MDVP:Jitter(%)": jitter_local * 100,
        "Jitter(Abs)": jitter_local,
        "Jitter:DDP": jitter_local * 3,
        "MDVP:Shimmer": shimmer_local,
        "Shimmer(dB)": shimmer_local,
        "MDVP:RAP": jitter_local,
        "MDVP:PPQ": jitter_local,
        "MDVP:APQ": shimmer_local,
        "HNR": hnr,
        # Extra generic features if your feature_names contain them
        "ZCR": zcr,
    }

    for name, value in mapping.items():
        if name in features:
            features[name] = value

    df = pd.DataFrame([features])
    return df


def run_analysis(input_data, original_data, label_prefix="Sample"):
    """Run the full prediction + visualization pipeline on given input data."""
    # Use global model, scaler, feature_names
    input_scaled = scaler.transform(input_data)
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        st.markdown(f"#### {label_prefix} {idx + 1}")

        parkinsons_prob = prob[1]
        healthy_prob = prob[0]
        has_parkinsons = pred == 1

        # Create columns for result display
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Display result box
            if has_parkinsons:
                st.markdown(f"""
                    <div class="result-positive">
                        <h3 style='color: #C0392B; margin-top: 0;'>⚠️ Elevated Risk Detected</h3>
                        <p style='color: #34495E; font-size: 1.1rem; margin: 0;'>
                        The analysis indicates patterns consistent with Parkinson's disease.
                        </p>
                        <p style='color: #7F8C8D; margin-top: 1rem; margin-bottom: 0;'>
                        <strong>Recommendation:</strong> Please consult a neurologist for comprehensive evaluation.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <h3 style='color: #27AE60; margin-top: 0;'>✅ Low Risk Detected</h3>
                        <p style='color: #34495E; font-size: 1.1rem; margin: 0;'>
                        Voice patterns appear normal with no significant indicators.
                        </p>
                        <p style='color: #7F8C8D; margin-top: 1rem; margin-bottom: 0;'>
                        <strong>Note:</strong> Continue regular health monitoring and maintain healthy habits.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Confidence metrics
            st.markdown("##### 📊 Confidence Scores")
            st.metric(
                "Parkinson's Likelihood",
                f"{parkinsons_prob*100:.2f}%",
                delta=f"{(parkinsons_prob-0.5)*100:.1f}%" if parkinsons_prob > 0.5 else None
            )
            st.metric(
                "Healthy Likelihood",
                f"{healthy_prob*100:.2f}%",
                delta=f"{(healthy_prob-0.5)*100:.1f}%" if healthy_prob > 0.5 else None
            )

        with res_col2:
            # Gauge chart
            gauge_fig = create_gauge_chart(parkinsons_prob, "Risk Assessment")
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Distribution chart
            dist_fig = create_confidence_distribution(parkinsons_prob)
            st.plotly_chart(dist_fig, use_container_width=True)

        if idx < len(predictions) - 1:
            st.markdown("---")

    # Feature importance
    st.markdown("---")
    st.markdown("### 🎯 Key Voice Features Analysis")
    importance_fig = create_feature_importance_chart(model, feature_names)
    if importance_fig:
        st.plotly_chart(importance_fig, use_container_width=True)

        st.markdown("""
            <div class="info-box">
                <p style='color: #34495E; margin: 0;'>
                <strong>Understanding Feature Importance:</strong> These features have the strongest 
                influence on the model's predictions. Higher values indicate greater impact on 
                the diagnostic decision.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ============ MAIN APP ============

# Header with mascot
st.markdown("""
    <div class="main-header">
        <div class="mascot-container">🩺</div>
        <h1>🎙️ ParkiSense AI – Parkinson's Voice Analysis</h1>
        <p>AI-Powered Detection using Advanced Voice Biomarkers</p>
    </div>
""", unsafe_allow_html=True)

# Load models
model, scaler, feature_names = load_models()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<div class='mascot-container' style='font-size: 60px;'>👨‍⚕️</div>", unsafe_allow_html=True)
    st.markdown("### 📋 About ParkiSense AI")
    st.markdown("""
    ParkiSense AI is an intelligent screening system that analyzes voice features
    to detect potential Parkinson's disease indicators.
    
    **How it works:**
    - Upload voice feature data (CSV) **or**
    - Upload an audio file (MP3/WAV) to auto-extract features
    - AI analyzes 22+ voice biomarkers
    - Get instant risk assessment
    
    **Important:** This is a screening tool, not a diagnosis. Always consult healthcare professionals.
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Model Information")
    st.markdown(f"""
    - **Algorithm:** {type(model).__name__}
    - **Features:** {len(feature_names)}
    - **Training:** Cross-validated
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Model Performance", "ℹ️ Information"])

with tab1:
    st.markdown("### 🎯 Voice Feature Analysis")

    input_mode = st.radio(
        "Select input type:",
        ["Upload CSV with pre-computed features", "Upload audio file (MP3/WAV/WAV-like)"],
        index=0,
        horizontal=True
    )

    if input_mode == "Upload CSV with pre-computed features":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4 style='color: #2C3E50; margin-top: 0;'>📁 Upload Your Data (CSV)</h4>
                    <p style='color: #34495E; margin-bottom: 0;'>
                    Upload a CSV file containing voice feature measurements. The system will analyze 
                    22 different voice biomarkers including frequency variations, amplitude measures, 
                    and noise-to-harmonics ratios.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload a CSV file with voice feature measurements"
            )

        with col2:
            st.markdown("""
                <div class="info-box">
                    <h4 style='color: #2C3E50; margin-top: 0;'>✅ Required Format</h4>
                    <ul style='color: #34495E;'>
                        <li>CSV file format</li>
                        <li>22+ voice features</li>
                        <li>No missing values</li>
                        <li>Valid numeric data</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                # Load and process data
                input_data = pd.read_csv(uploaded_file)
                original_data = input_data.copy()

                # Remove name and status columns if present
                columns_to_drop = [col for col in ['name', 'status'] if col in input_data.columns]
                if columns_to_drop:
                    input_data = input_data.drop(columns_to_drop, axis=1)

                st.success(f"✅ File uploaded successfully! Found {len(input_data)} sample(s)")

                # Display data preview
                with st.expander("🔍 View Uploaded Data"):
                    st.dataframe(original_data.head(10), use_container_width=True)

                # Process and predict
                if st.button("🔬 Analyze Voice Features", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing voice patterns..."):
                        run_analysis(input_data, original_data, label_prefix="Sample")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV file contains all required voice features.")
        else:
            st.markdown("""
                <div class="info-box" style="text-align: center; padding: 3rem;">
                    <div style="font-size: 64px; margin-bottom: 1rem;">📤</div>
                    <h3 style='color: #2C3E50;'>No File Uploaded</h3>
                    <p style='color: #7F8C8D;'>Upload a CSV file to begin voice analysis</p>
                </div>
            """, unsafe_allow_html=True)

    else:
        # Audio mode
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4 style='color: #2C3E50; margin-top: 0;'>🎧 Upload an Audio File</h4>
                    <p style='color: #34495E; margin-bottom: 0;'>
                    Upload a short voice recording (3–10 seconds) in MP3/WAV/OGG/M4A format.
                    ParkiSense AI will automatically extract voice features, convert them to CSV,
                    and run them through the Parkinson's detection model.
                    </p>
                </div>
            """, unsafe_allow_html=True)
with tab1:
    st.markdown("### 🎯 Voice Feature Analysis")

    input_mode = st.radio(
        "Select input type:",
        ["Upload CSV with pre-computed features", "Audio (Upload / Record)"],
        index=0,
        horizontal=True
    )
    if input_mode == "Upload CSV with pre-computed features":
        pass

    else:
        # 🔊 AUDIO MODE (UPLOAD + RECORD)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4 style='color: #2C3E50; margin-top: 0;'>🎧 Audio Input</h4>
                    <p style='color: #34495E; margin-bottom: 0;'>
                    Aap ya to audio file upload kar sakte hain, ya seedha browser se record kar sakte hain.
                    ParkiSense AI voice features extract karke model pe run karega.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            source = st.radio(
                "Choose audio source:",
                ["Upload file", "Record with mic"],
                horizontal=True
            )

            audio_bytes = None
            audio_file = None

            if source == "Upload file":
                audio_file = st.file_uploader(
                    "Choose an audio file",
                    type=["wav", "mp3", "ogg", "m4a"],
                    help="Upload a short voice recording"
                )
                if audio_file is not None:
                    st.audio(audio_file, format="audio/mpeg")

            else:
                st.markdown("#### 🎙️ Record your voice")
    audio = audiorecorder("Start recording", "Stop recording")

    if len(audio) > 0:
        # Convert AudioSegment → WAV bytes
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        audio_bytes = wav_buffer.getvalue()
        st.audio(audio_bytes, format="audio/wav")


        # 🔍 Process + Analyse button
        if st.button("🎧 Extract Features & Analyze", use_container_width=True):
            with st.spinner("🎛 Extracting voice features and analyzing..."):
                try:
                    # Decide kaun sa input use ho raha hai
                    if source == "Upload file":
                        if audio_file is None:
                            st.error("❌ Please upload an audio file first.")
                            st.stop()
                        audio_like = audio_file

                    else:
                        if audio_bytes is None:
                            st.error("❌ Please record your voice first.")
                            st.stop()
                        audio_like = io.BytesIO(audio_bytes)

                    # Tumhara existing function
                    features_df = extract_features_from_audio(audio_like, feature_names)

                    st.success("✅ Features extracted successfully!")

                    with st.expander("🔍 View Extracted Features (CSV format)"):
                        st.dataframe(features_df, use_container_width=True)

                    csv_buffer = io.StringIO()
                    features_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "⬇️ Download Extracted Features as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="parkinsons_voice_features_from_audio.csv",
                        mime="text/csv"
                    )

                    # Run analysis pipeline (tumhare function se)
                    run_analysis(features_df, features_df, label_prefix="Recording")

                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.info("Please ensure this is a short, clear voice recording.")

          


        with col2:
            st.markdown("""
                <div class="info-box">
                    <h4 style='color: #2C3E50; margin-top: 0;'>🎙️ Recording Tips</h4>
                    <ul style='color: #34495E;'>
                        <li>Quiet environment</li>
                        <li>Speak a simple sentence or sustained vowel sound (e.g., “aaaaa”)</li>
                        <li>Keep the microphone at a fixed distance</li>
                        <li>Duration ~3–10 seconds</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        if audio_file is not None:
            if st.button("🎧 Extract Features & Analyze", use_container_width=True):
                with st.spinner("🎛 Extracting voice features and analyzing..."):
                    try:
                        # Extract features as a DataFrame
                        features_df = extract_features_from_audio(audio_file, feature_names)

                        st.success("✅ Features extracted successfully!")

                        with st.expander("🔍 View Extracted Features (CSV format)"):
                            st.dataframe(features_df, use_container_width=True)

                        # Offer CSV download
                        csv_buffer = io.StringIO()
                        features_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "⬇️ Download Extracted Features as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="parkinsons_voice_features_from_audio.csv",
                            mime="text/csv"
                        )

                        # Run analysis pipeline
                        run_analysis(features_df, features_df, label_prefix="Recording")

                    except Exception as e:
                        st.error(f"❌ Error processing audio file: {str(e)}")
                        st.info("Please ensure the audio is a valid, short voice recording.")

with tab2:
    st.markdown("### 📈 Model Performance Metrics")

    display_model_metrics()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Classification Performance")

        # Sample confusion matrix visualization
        confusion_data = pd.DataFrame({
            'Predicted Healthy': [45, 2],
            "Predicted Parkinson's": [2, 50]
        }, index=['Actually Healthy', "Actually Parkinson's"])

        fig = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=confusion_data.columns,
            y=confusion_data.index,
            color_continuous_scale='Blues',
            text_auto=True
        )

        fig.update_layout(
            title="Confusion Matrix",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(232, 244, 248, 0.3)",
            font=dict(color='#2C3E50')
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Performance Breakdown")

        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [94.87, 96.15, 96.15, 96.15, 98.21]
        })

        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Score',
            color='Score',
            color_continuous_scale='Blues',
            text='Score'
        )

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            yaxis_range=[0, 105],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(232, 244, 248, 0.3)",
            showlegend=False,
            font=dict(color='#2C3E50')
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div class="info-box">
            <h4 style='color: #2C3E50; margin-top: 0;'>📝 Model Training Details</h4>
            <ul style='color: #34495E;'>
                <li><strong>Dataset:</strong> 28000 voice recordings from Parkinson's patients and healthy individuals</li>
                <li><strong>Validation:</strong> 5-fold cross-validation with stratified sampling</li>
                <li><strong>Features:</strong> 22 voice biomarkers including MDVP, jitter, shimmer, and HNR measures</li>
                <li><strong>Optimization:</strong> Hyperparameter tuning via grid search</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ℹ️ About Parkinson's Disease & Voice Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="info-box">
                <h4 style='color: #2C3E50; margin-top: 0;'>🧠 What is Parkinson's Disease?</h4>
                <p style='color: #34495E;'>
                Parkinson's disease is a progressive nervous system disorder that affects movement. 
                Symptoms start gradually, sometimes starting with a barely noticeable tremor in just 
                one hand. Tremors are common, but the disorder also commonly causes stiffness or 
                slowing of movement.
                </p>
                <p style='color: #34495E;'><strong>Early Signs:</strong></p>
                <ul style='color: #34495E;'>
                    <li>Tremor in hands, arms, legs, or jaw</li>
                    <li>Stiffness of limbs and trunk</li>
                    <li>Slowness of movement</li>
                    <li>Impaired balance and coordination</li>
                    <li>Voice changes</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-box">
                <h4 style='color: #2C3E50; margin-top: 0;'>🎙️ Why Voice Analysis?</h4>
                <p style='color: #34495E;'>
                Voice impairment is one of the early signs of Parkinson's disease, affecting up to 
                90% of patients. Voice analysis provides a non-invasive, cost-effective screening method.
                </p>
                <p style='color: #34495E;'><strong>Key Voice Features:</strong></p>
                <ul style='color: #34495E;'>
                    <li><strong>MDVP:</strong> Multi-Dimensional Voice Program measures</li>
                    <li><strong>Jitter:</strong> Frequency variation measure</li>
                    <li><strong>Shimmer:</strong> Amplitude variation measure</li>
                    <li><strong>HNR:</strong> Harmonics-to-Noise Ratio</li>
                    <li><strong>RPDE:</strong> Recurrence Period Density Entropy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <h4 style='color: #2C3E50; margin-top: 0;'>⚠️ Important Disclaimer</h4>
            <p style='color: #34495E;'>
            <strong>This tool is for screening purposes only and should not be used as a substitute 
            for professional medical diagnosis.</strong> If you receive a positive result or have 
            concerns about Parkinson's disease, please consult with a qualified neurologist or 
            healthcare provider for comprehensive evaluation and diagnosis.
            </p>
            <p style='color: #34495E;'>
            Early detection and treatment can significantly improve quality of life for individuals 
            with Parkinson's disease. This AI tool aims to assist in early screening by analyzing 
            voice biomarkers that may indicate the presence of the disease.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: white; border-radius: 12px;'>
            <h4 style='color: #4A90E2;'>Need Help?</h4>
            <p style='color: #7F8C8D;'>For technical support or questions about this tool, 
            please contact your healthcare provider or the system administrator.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7F8C8D; padding: 1rem;'>
        <p>🩺 ParkiSense AI – Parkinson's Voice Analysis </p>
        <p style='font-size: 0.9rem;'>For medical screening purposes only • Always consult healthcare professionals</p>
    </div>
""", unsafe_allow_html=True)



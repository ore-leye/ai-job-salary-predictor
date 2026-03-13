import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG + STYLING
# -------------------------------------------------
st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    .big-title {font-size: 42px !important; font-weight: bold; color: #2563EB; text-align: center; margin-bottom: 10px;}
    .card {background-color: #F9FAFB; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 24px;}
    .success-card {background-color: #EFF6FF; border-left: 5px solid #2563EB; padding: 16px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">💼 AI Job Salary Predictor</p>', unsafe_allow_html=True)
st.caption("Predict AI job salaries • Classify high/low • Explore market insights")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "salary_data.csv"
MODELS_DIR = BASE_DIR / "models"
REG_PATH = MODELS_DIR / "models.pkl"
CLS_PATH = MODELS_DIR / "models_2.pkl"
CLASSES_PATH = MODELS_DIR / "classes.json"
MARKET_STATS_PATH = MODELS_DIR / "salary_market_stats.json"

# -------------------------------------------------
# LOAD MODELS – STRONGEST EXTRACTION
# -------------------------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error("salary_data.csv not found.")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_all_models():
    missing = [p.name for p in [REG_PATH, CLS_PATH, CLASSES_PATH, MARKET_STATS_PATH] if not p.exists()]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        st.stop()

    reg_content = joblib.load(REG_PATH)
    cls_content = joblib.load(CLS_PATH)

    # === STRONG EXTRACTION LOGIC ===
    def extract_model(content):
        if not isinstance(content, dict):
            return content
        possible_keys = ["pipeline", "model", "reg", "regression", "estimator", "lgbm", 
                        "classifier", "clf", "regressor", "lgb", "lightgbm", "main_model", 
                        "best_model", "lgbm_model", "trained_model", "best_estimator"]
        for key in possible_keys:
            if key in content:
                return content[key]
        # Last resort: find any object that has .predict method
        for value in content.values():
            if hasattr(value, "predict"):
                return value
        return next(iter(content.values())) if len(content) == 1 else content

    reg_pipeline = extract_model(reg_content)
    cls_pipeline = extract_model(cls_content)

    with open(CLASSES_PATH, 'r') as f:
        classes = json.load(f)
    with open(MARKET_STATS_PATH, 'r') as f:
        market_stats = json.load(f)

    return reg_pipeline, cls_pipeline, classes, market_stats

df = load_dataset()
reg_pipeline, cls_pipeline, classes_dict, market_stats = load_all_models()

st.success("Models & data loaded successfully!", icon="✅")

# -------------------------------------------------
# SKILLS LIST
# -------------------------------------------------
SKILL_LIST = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PyTorch", "TensorFlow", "AWS", "Azure", "Docker", "Kubernetes", "Git", "Hadoop",
    "Spark", "Tableau", "Power BI", "Scala", "Java", "Linux",
    "MLOps", "R", "GCP", "Data Visualization", "Statistics", "Mathematics"
]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
page = st.sidebar.radio("Section", ["Single Prediction", "Bulk Prediction", "Market Insights"])

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
if page == "Single Prediction":
    # (Your full Single Prediction code stays exactly the same as before)
    # ... [paste your existing Single Prediction block here - no changes needed] ...

# -------------------------------------------------
# BULK PREDICTION (uses "Upload")
# -------------------------------------------------
elif page == "Bulk Prediction":
    st.header("Bulk Salary Prediction")

    uploaded_file = st.file_uploader("Upload CSV with job data", type=["csv"])

    if uploaded_file is not None:
        # (Your existing bulk code stays the same)
        # ... [paste your existing Bulk Prediction block here] ...

# -------------------------------------------------
# MARKET INSIGHTS
# -------------------------------------------------
elif page == "Market Insights":
    st.header("Market Insights Dashboard")
    # (Your existing Market Insights code stays the same)
    # ... [paste your existing Market Insights block here] ...

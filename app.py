import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG + STYLING (blue + neutral gray)
# -------------------------------------------------
st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    .big-title {
        font-size: 42px !important;
        font-weight: bold;
        color: #2563EB;
        text-align: center;
        margin-bottom: 10px;
    }
    .card {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 24px;
    }
    .success-card {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
    }
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
# LOAD RESOURCES – Enhanced extraction
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

    # Extract actual model from dict (expanded matching)
    possible_keys = [
        "pipeline", "model", "reg", "regression", "estimator", "lgbm", "classifier",
        "clf", "regressor", "lgb", "lightgbm", "main_model", "best_model", "lgbm_model",
        "trained_model", "best_estimator", "final_model"
    ]

    reg_pipeline = None
    if isinstance(reg_content, dict):
        for key in possible_keys:
            if key in reg_content:
                reg_pipeline = reg_content[key]
                break
        if reg_pipeline is None and len(reg_content) == 1:
            reg_pipeline = next(iter(reg_content.values()))
        elif reg_pipeline is None:
            # Last resort: try the first callable value (model-like)
            reg_pipeline = next((v for v in reg_content.values() if hasattr(v, 'predict')), None)

    if reg_pipeline is None:
        reg_pipeline = reg_content

    cls_pipeline = None
    if isinstance(cls_content, dict):
        for key in possible_keys:
            if key in cls_content:
                cls_pipeline = cls_content[key]
                break
        if cls_pipeline is None and len(cls_content) == 1:
            cls_pipeline = next(iter(cls_content.values()))
        elif cls_pipeline is None:
            cls_pipeline = next((v for v in cls_content.values() if hasattr(v, 'predict')), None)

    if cls_pipeline is None:
        cls_pipeline = cls_content

    with open(CLASSES_PATH, 'r') as f:
        classes = json.load(f)
    with open(MARKET_STATS_PATH, 'r') as f:
        market_stats = json.load(f)

    return reg_pipeline, cls_pipeline, classes, market_stats

df = load_dataset()
reg_pipeline, cls_pipeline, classes_dict, market_stats = load_all_models()

st.success("Models & data loaded successfully!", icon="✅")

# -------------------------------------------------
# SKILLS LIST (includes all from your errors)
# -------------------------------------------------
SKILL_LIST = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PyTorch", "TensorFlow", "AWS", "Azure", "Docker", "Kubernetes", "Git", "Hadoop",
    "Spark", "Tableau", "Power BI", "Scala", "Java", "Linux",
    "MLOps", "R", "GCP", "Data Visualization", "Statistics", "Mathematics"
]

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio("Section", ["Single Prediction", "Bulk Prediction", "Market Insights"])

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
if page == "Single Prediction":
    st.header("Predict Salary for a New Job Posting")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Job Title", sorted(df["job_title"].dropna().unique()))
        experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
        years_experience = st.slider("Years of Experience", 0, 20, 4)
        remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 50)

    with col2:
        education = st.selectbox("Education Required", ["Bachelor", "Master", "PhD", "Associate", "Other"])
        company_size = st.selectbox("Company Size", ["S", "M", "L"])
        industry = st.selectbox("Industry", sorted(df["industry"].dropna().unique()))
        skills = st.multiselect("Required Skills", SKILL_LIST, default=["Python", "SQL"])

    col3, col4 = st.columns(2)
    with col3:
        job_desc_len = st.number_input("Job Description Length (chars)", 100, 5000, 1200)
        benefits_score = st.slider("Benefits Score (0–10)", 0.0, 10.0, 6.5)
    with col4:
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
        company_location = st.selectbox("Company Location", ["United States", "India", "Canada", "Germany", "United Kingdom"])
        employee_residence = st.selectbox("Employee Residence", ["United States", "India", "Canada", "Germany", "United Kingdom"])
        job_demand_period = st.number_input("Job Demand Period (days)", 0, 180, 30)

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                input_data = {
                    "job_title": job_title,
                    "experience_level": experience_level,
                    "years_experience": years_experience,
                    "remote_ratio": remote_ratio,
                    "education_required": education,
                    "company_size": company_size,
                    "industry": industry,
                    "job_description_length": job_desc_len,
                    "benefits_score": benefits_score,
                    "employment_type": employment_type,
                    "company_location": company_location,
                    "employee_residence": employee_residence,
                    "job_demand_period": job_demand_period,
                    "company_name": "Unknown Company",
                }

                for skill in SKILL_LIST:
                    input_data[f"skill_{skill}"] = 1 if skill in skills else 0

                X_input = pd.DataFrame([input_data])

                salary = float(reg_pipeline.predict(X_input)[0])
                category_idx = int(cls_pipeline.predict(X_input)[0])
                category_name = classes_dict[str(category_idx)]

                st.markdown(f"""
                <div class="success-card">
                    <h2 style="margin:0; color:#065f46;">Predicted Salary: ${salary:,.0f}</h2>
                    <p style="margin:8px 0 0; font-size:1.1rem;"><strong>Category:</strong> {category_name}</p>
                </div>
                """, unsafe_allow_html=True)

                # Market Position
                st.subheader("Market Position")
                median = market_stats["median"]
                p25, p75 = market_stats["p25"], market_stats["p75"]

                if salary < p25:
                    pos = "Below Market Average 🔴"
                elif salary < median:
                    pos = "Lower Half 🟠"
                elif salary <= p75:
                    pos = "Around Market Average 🟢"
                else:
                    pos = "Above Average / Top Tier 🟡"

                st.info(f"**{pos}** (based on current market data)")

                # Career Growth
                st.subheader("5-Year Salary Growth Projection")
                growth_rates = {"EN": 0.085, "MI": 0.065, "SE": 0.045, "EX": 0.035}
                rate = growth_rates.get(experience_level, 0.06)

                years = list(range(6))
                salaries_proj = [salary]
                current = salary
                for _ in range(5):
                    current *= (1 + rate)
                    salaries_proj.append(round(current, -2))

                proj_df = pd.DataFrame({"Year": years, "Projected Salary": [f"${s:,.0f}" for s in salaries_proj]})
                st.table(proj_df.style.set_properties(**{'text-align': 'center'}))

                fig = go.Figure(go.Scatter(
                    x=years, y=salaries_proj,
                    mode='lines+markers',
                    line=dict(color='#2563EB', width=3.5),
                    marker=dict(size=9)
                ))
                fig.update_layout(
                    title="Projected Salary Growth",
                    xaxis_title="Years from Now",
                    yaxis_title="Salary (USD)",
                    yaxis_tickprefix="$", yaxis_tickformat=",",
                    height=380
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Growth rate: **{rate:.1%}** per year (typical for {experience_level})")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# -------------------------------------------------
# BULK PREDICTION – Uses "Upload" instead of "Browse"
# -------------------------------------------------
elif page == "Bulk Prediction":
    st.header("Bulk Salary Prediction")

    uploaded_file = st.file_uploader("Upload CSV with job data (same columns as training)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df_upload.head())

            # Add missing columns with defaults
            default_values = {
                "employment_type": "Full-time",
                "company_location": "United States",
                "employee_residence": "United States",
                "job_demand_period": 30,
                "company_name": "Unknown Company",
            }
            for col, val in default_values.items():
                if col not in df_upload.columns:
                    df_upload[col] = val

            # Add all skill columns (0 by default)
            for skill in SKILL_LIST:
                skill_col = f"skill_{skill}"
                if skill_col not in df_upload.columns:
                    df_upload[skill_col] = 0

            # Predict
            with st.spinner("Predicting batch..."):
                salaries = reg_pipeline.predict(df_upload)
                categories = cls_pipeline.predict(df_upload)
                df_results = df_upload.copy()
                df_results["Predicted Salary"] = salaries.round(0).astype(int)
                df_results["Salary Category"] = [classes_dict[str(int(c))] for c in categories]

            st.success("Batch prediction complete!")
            st.dataframe(df_results.head(10))

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing bulk file: {str(e)}")

# -------------------------------------------------
# MARKET INSIGHTS
# -------------------------------------------------
elif page == "Market Insights":
    st.header("Market Insights Dashboard")

    if df.empty:
        st.warning("No data loaded for insights.")
    else:
        # Basic stats
        col1, col2, col3 = st.columns(3)
        salary_col = "salary_usd" if "salary_usd" in df.columns else df.filter(like="salary").columns[0] if any("salary" in c for c in df.columns) else None

        if salary_col:
            avg = df[salary_col].mean()
            med = df[salary_col].median()
            max_s = df[salary_col].max()

            col1.metric("Average Salary", f"${avg:,.0f}")
            col2.metric("Median Salary", f"${med:,.0f}")
            col3.metric("Highest Salary", f"${max_s:,.0f}")

            # Salary distribution
            st.subheader("Salary Distribution")
            fig_dist = px.histogram(df, x=salary_col, nbins=30, title="Salary Distribution")
            fig_dist.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Industry comparison
            st.subheader("Salary by Industry")
            if "industry" in df.columns:
                industry_stats = df.groupby("industry")[salary_col].agg(["mean", "count"]).sort_values("mean", ascending=False)
                fig_ind = px.bar(industry_stats.reset_index(), x="industry", y="mean", 
                                 title="Average Salary by Industry",
                                 labels={"mean": "Average Salary", "industry": "Industry"})
                fig_ind.update_yaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.info("Industry column not found for comparison.")

        else:
            st.warning("No salary column found in dataset for statistics.")
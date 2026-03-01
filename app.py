# ============================================
# Healthcare Risk Classification - Optimized
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import io, os, hashlib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Healthcare Risk Management System",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CSS — Reference UI (dark but readable, green accent, mono font)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --surface2: #1f2335;
    --border:   #252a3a;
    --accent:   #00e5a0;
    --accent2:  #ff6b35;
    --accent3:  #5b8cff;
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --danger:   #ff4560;
    --warning:  #f59e0b;
}

/* ── Base ── */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div,
.main > div { background: var(--bg) !important; color: var(--text) !important; }

html, body, [class*="css"] { font-family: 'DM Mono', monospace !important; }
.block-container { padding-top: 1.8rem !important; max-width: 1400px; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }
p, li, span { color: var(--text) !important; }
strong, b { color: var(--text) !important; }
label { color: var(--muted) !important; font-size: 12px !important; }

/* ── Header bar ── */
.main-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 20px 28px;
    border-radius: 10px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.main-header::after {
    content: '';
    position: absolute; top: 0; right: 0; width: 160px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.03));
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
    font-size: 22px !important; font-weight: 800 !important;
    margin: 0 !important; letter-spacing: -0.2px;
}
.main-header p { color: var(--muted) !important; margin: 4px 0 0 !important; font-size: 12px !important; }
.eyebrow {
    font-size: 10px; font-weight: 500; letter-spacing: 0.2em;
    color: var(--accent) !important; text-transform: uppercase; margin-bottom: 5px;
    font-family: 'DM Mono', monospace;
}

/* ── KPI cards ── */
[data-testid="stMetric"],
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover,
[data-testid="metric-container"]:hover {
    border-color: var(--accent) !important;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"],
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important; font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Risk badges ── */
.risk-badge {
    padding: 15px 24px; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 18px;
    text-align: center; letter-spacing: 0.4px; margin-bottom: 18px; border: 1px solid;
}
.risk-low      { background: rgba(0,229,160,0.10); border-color: rgba(0,229,160,0.35);  color: #00e5a0; }
.risk-medium   { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.35); color: #f59e0b; }
.risk-high     { background: rgba(255,107,53,0.10); border-color: rgba(255,107,53,0.35); color: #ff6b35; }
.risk-critical { background: rgba(255,69,96,0.10);  border-color: rgba(255,69,96,0.35);  color: #ff4560; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"],
[data-testid="stTabs"] button {
    height: 42px;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 10px 20px;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; font-size: 12px !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover,
[data-testid="stTabs"] button:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"],
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; font-size: 13px !important;
    letter-spacing: 0.06em !important; padding: 10px 22px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #1aedac !important;
    box-shadow: 0 3px 14px rgba(0,229,160,0.28) !important;
    transform: translateY(-1px) !important;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px; font-weight: 700; color: var(--text);
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ── Pipeline steps ── */
.pipeline-step {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 7px 13px; margin-bottom: 5px;
    font-size: 12px; font-family: 'DM Mono', monospace;
    color: var(--text);
}
.pipeline-step strong { color: var(--accent) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; overflow: hidden;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: var(--text) !important; background: var(--surface) !important;
    letter-spacing: 0.04em;
}
[data-testid="stExpander"] > div > div { background: var(--surface) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important; border-left-width: 3px !important;
    font-size: 13px !important; font-family: 'DM Mono', monospace !important;
    background: var(--surface) !important; color: var(--text) !important;
}

/* ── Inputs & selects ── */
/* ── Aggressive Dropdown Menu (Popover) Overrides ── */
[data-baseweb="popover"],
[data-baseweb="menu"],
div[role="listbox"],
ul[role="listbox"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

[data-baseweb="menu"] li,
div[role="listbox"] li,
ul[role="listbox"] li {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    background-color: transparent !important;
}

[data-baseweb="menu"] li:hover,
div[role="listbox"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: rgba(0, 229, 160, 0.1) !important;
    color: var(--accent) !important;
}

/* ── Aggressive File Uploader Overrides ── */
[data-testid="stFileUploadDropzone"] {
    background-color: var(--surface) !important;
}

/* Forces the "Drag and drop file here" text to be visible */
[data-testid="stFileUploadDropzone"] [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploadDropzone"] span {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Forces the "Limit 200MB" text to be muted */
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] div[data-testid="stText"] {
    color: var(--muted) !important;
}

/* Ensures the "Browse files" button remains readable */
[data-testid="stFileUploadDropzone"] button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stFileUploadDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid rgba(0,229,160,0.4) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; font-size: 12px !important;
    letter-spacing: 0.05em !important; transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,229,160,0.07) !important;
    border-color: var(--accent) !important;
}

/* ── HR ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 22px 0 !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
}

/* ── Code ── */
[data-testid="stCode"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Chart colours — vivid on dark background ──
RISK_COLOR_MAP = {
    'LOW':      '#00e5a0',
    'MEDIUM':   '#f59e0b',
    'HIGH':     '#ff6b35',
    'CRITICAL': '#ff4560'
}
PLOT_BG    = "#1a1d27"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "#252a3a"
FONT_COLOR = "#6b7280"
TEXT_COLOR = "#e2df3f"


# ============================================
# LOGIN
# ============================================
def login():
    st.markdown("""
        <div style='text-align:center; margin-top:60px;'>
            <div style='width:64px; height:64px; background:#1a1d27;
                        border:1px solid rgba(0,229,160,0.35); border-radius:14px;
                        margin:0 auto 16px; display:flex; align-items:center;
                        justify-content:center; font-size:30px;'>🏥</div>
            <div style='font-size:10px; font-weight:500; letter-spacing:0.2em;
                        color:#00e5a0; text-transform:uppercase; margin-bottom:12px;
                        font-family:"DM Mono",monospace;'>Healthcare Risk Management</div>
            <h1 style='font-family:"Syne",sans-serif; color:#e8eaf0;
                       font-size:28px; font-weight:800; margin:0;'>Clinical Intelligence Portal</h1>
            <p style='color:#6b7280; font-size:12px; margin-top:8px;
                      font-family:"DM Mono",monospace;'>Authorized Personnel Only</p>
        </div>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
            <div style='background:#1a1d27; border:1px solid #252a3a;
                        border-radius:12px; padding:28px 24px; margin-top:18px;'>
        """, unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="username")
        password = st.text_input("Password", type="password", placeholder="password")
        if st.button("→  Login", use_container_width=True):
            if username.strip() == "admin" and password.strip() == "SecureHealth2026!":
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("demo  ·  admin / SecureHealth2026!")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()

# ============================================
# SESSION STATE
# ============================================
_defaults = {
    "df": None, "model": None, "preprocessor": None,
    "label_encoder": None, "feature_names": None,
    "model_trained": False, "last_file_hash": None,
    "shap_explainer": None
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class='main-header'>
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='width:44px; height:44px; background:rgba(0,229,160,0.09);
                    border:1px solid rgba(0,229,160,0.25); border-radius:10px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:22px; flex-shrink:0;'>🏥</div>
        <div>
            <div class='eyebrow'>Clinical Intelligence Platform</div>
            <h1>AI Patient Risk Analytics</h1>
            <p>ML-powered predictive healthcare framework — upload your dataset to begin</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================
# SHARED: ENGINEERED FEATURES HELPER
# BUG FIX — called in all 3 places: cleaning, DB lookup, manual assessment
# ============================================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotently add all engineered columns to any dataframe."""
    df = df.copy()
    if "Systolic_BP" in df.columns and "Diastolic_BP" in df.columns and "MAP" not in df.columns:
        df["MAP"] = ((df["Systolic_BP"] + 2 * df["Diastolic_BP"]) / 3).round(2)
    if "BMI" in df.columns and "Age" in df.columns and "BMI_Age_Index" not in df.columns:
        df["BMI_Age_Index"] = (df["BMI"] * df["Age"] / 100).round(3)
    if "HbA1c_percent" in df.columns and "BMI" in df.columns and "Metabolic_Risk_Score" not in df.columns:
        df["Metabolic_Risk_Score"] = (df["HbA1c_percent"] * 0.6 + df["BMI"] * 0.4).round(3)
    if "Physical_Activity_hours_per_week" in df.columns and "Daily_Steps" in df.columns and "Activity_Score" not in df.columns:
        df["Activity_Score"] = (df["Physical_Activity_hours_per_week"] * 200 + df["Daily_Steps"]).round(0)
    if "Sleep_Hours_per_night" in df.columns and "Sleep_Quality" not in df.columns:
        df["Sleep_Quality"] = np.where(df["Sleep_Hours_per_night"].between(7, 9), 1, 0)
    return df


# ============================================
# HELPERS
# ============================================
def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def pipeline_step(icon, label, detail):
    st.markdown(
        f"<div class='pipeline-step'>{icon} <strong>{label}:</strong> {detail}</div>",
        unsafe_allow_html=True
    )

def render_risk_badge(risk_level, prefix="Assessment Result: "):
    rl  = risk_level.upper()
    css = ("risk-critical" if "CRITICAL" in rl else
           "risk-high"     if "HIGH"     in rl else
           "risk-medium"   if "MEDIUM"   in rl else "risk-low")
    st.markdown(
        f"<div class='risk-badge {css}'>🩺 {prefix}{rl} RISK</div>",
        unsafe_allow_html=True
    )

def plotly_layout(**kwargs):
    return dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_family="DM Mono", font_color=TEXT_COLOR,
        title_font_family="Syne", title_font_color=TEXT_COLOR,
        title_font_size=14,
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                   zerolinecolor=GRID_COLOR, tickfont_color=FONT_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                   zerolinecolor=GRID_COLOR, tickfont_color=FONT_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_COLOR),
        **kwargs
    )


# ============================================
# CACHED: CLEAN + FEATURE ENGINEERING
# ============================================
@st.cache_data(show_spinner=False)
def auto_clean_and_engineer(raw_bytes: bytes) -> tuple[pd.DataFrame, list]:
    import io as _io
    try:
        df = pd.read_excel(_io.BytesIO(raw_bytes))
    except Exception:
        df = pd.read_csv(_io.BytesIO(raw_bytes))

    log = []
    df.columns = df.columns.str.strip()

    alias_map = {
        "patient_id":"Patient_ID","id":"Patient_ID","age":"Age","bmi":"BMI",
        "gender":"Gender","sex":"Gender","systolic_bp":"Systolic_BP","systolic":"Systolic_BP",
        "diastolic_bp":"Diastolic_BP","diastolic":"Diastolic_BP",
        "cholesterol":"Cholesterol_mg_dL","cholesterol_mg_dl":"Cholesterol_mg_dL",
        "hba1c":"HbA1c_percent","hba1c_percent":"HbA1c_percent",
        "smoking_status":"Smoking_Status","smoking":"Smoking_Status",
        "alcohol_consumption_per_week":"Alcohol_Consumption_per_week","alcohol":"Alcohol_Consumption_per_week",
        "physical_activity_hours_per_week":"Physical_Activity_hours_per_week","physical_activity":"Physical_Activity_hours_per_week",
        "sleep_hours_per_night":"Sleep_Hours_per_night","sleep_hours":"Sleep_Hours_per_night","sleep":"Sleep_Hours_per_night",
        "avg_heart_rate":"Avg_Heart_Rate","heart_rate":"Avg_Heart_Rate",
        "daily_steps":"Daily_Steps","steps":"Daily_Steps",
        "family_history_diabetes":"Family_History_Diabetes",
        "family_history_heart_disease":"Family_History_Heart_Disease",
        "risk_category":"Risk_Category","risk":"Risk_Category"
    }
    renamed = {c: alias_map[c.lower().replace(" ","_")]
               for c in df.columns
               if c.lower().replace(" ","_") in alias_map
               and c != alias_map[c.lower().replace(" ","_")]}
    if renamed:
        df.rename(columns=renamed, inplace=True)
        log.append(("🔄","Column Renaming",f"Renamed {len(renamed)} columns"))

    if "Risk_Category" in df.columns:
        df["Risk_Category"] = df["Risk_Category"].astype(str).str.strip().str.upper()
        df["Risk_Category"] = df["Risk_Category"].replace({
            "LO":"LOW","MED":"MEDIUM","HI":"HIGH","CRIT":"CRITICAL",
            "LOW RISK":"LOW","MEDIUM RISK":"MEDIUM","HIGH RISK":"HIGH","CRITICAL RISK":"CRITICAL"
        })
        log.append(("✅","Risk Labels","Standardized to LOW / MEDIUM / HIGH / CRITICAL"))

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
        df["Gender"] = df["Gender"].replace({"M":"Male","F":"Female","0":"Male","1":"Female"})
        log.append(("✅","Gender","Standardized to Male / Female"))

    if "Smoking_Status" in df.columns:
        df["Smoking_Status"] = df["Smoking_Status"].astype(str).str.strip().str.title()
        df["Smoking_Status"] = df["Smoking_Status"].replace({
            "Non Smoker":"Non-Smoker","Nonsmoker":"Non-Smoker",
            "Former":"Former Smoker","Current":"Current Smoker",
            "Yes":"Current Smoker","No":"Non-Smoker"
        })
        log.append(("✅","Smoking Status","Standardized labels"))

    num_cols = ["Age","BMI","Systolic_BP","Diastolic_BP","Cholesterol_mg_dL","HbA1c_percent",
                "Alcohol_Consumption_per_week","Physical_Activity_hours_per_week",
                "Sleep_Hours_per_night","Avg_Heart_Rate","Daily_Steps",
                "Family_History_Diabetes","Family_History_Heart_Disease"]
    coerced = [c for c in num_cols if c in df.columns and str(df[c].dtype) == "object"]
    for c in coerced:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if coerced:
        log.append(("🔢","Type Coercion",f"Converted: {', '.join(coerced)}"))

    missing = int(df.isnull().sum().sum())
    if missing > 0:
        for c in df.columns:
            if df[c].isnull().sum() > 0:
                df[c].fillna(
                    df[c].median() if df[c].dtype in ["float64","int64"]
                    else (df[c].mode()[0] if not df[c].mode().empty else "Unknown"),
                    inplace=True
                )
        log.append(("🩹","Missing Values",f"Imputed {missing} values"))
    else:
        log.append(("✅","Missing Values","None detected"))

    dup = int(df.duplicated().sum())
    if dup > 0:
        df.drop_duplicates(inplace=True)
        log.append(("🗑️","Duplicates",f"Removed {dup} rows"))
    else:
        log.append(("✅","Duplicates","None found"))

    clip_cols = ["BMI","Systolic_BP","Diastolic_BP","Cholesterol_mg_dL",
                 "HbA1c_percent","Avg_Heart_Rate","Daily_Steps"]
    n_out = 0
    for c in clip_cols:
        if c in df.columns:
            q1, q3 = df[c].quantile(0.01), df[c].quantile(0.99)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            n_out += int(((df[c] < lo) | (df[c] > hi)).sum())
            df[c] = df[c].clip(lo, hi)
    log.append(("📊","Outliers",f"Clipped {n_out} extreme values") if n_out > 0
               else ("✅","Outliers","None detected"))

    if "Patient_ID" not in df.columns:
        df.insert(0, "Patient_ID", [f"PT{str(i+1).zfill(4)}" for i in range(len(df))])
        log.append(("🆔","Patient IDs","Auto-generated"))

    # Feature engineering via shared helper
    df = add_engineered_features(df)
    fe_cols = [c for c in ["MAP","BMI_Age_Index","Metabolic_Risk_Score","Activity_Score","Sleep_Quality"]
               if c in df.columns]
    if fe_cols:
        log.append(("⚗️","Feature Engineering",f"Created: {', '.join(fe_cols)}"))

    return df, log


# ============================================
# CACHED: MODEL TRAINING
# ============================================
@st.cache_resource(show_spinner=False)
def train_model_cached(df_hash: str, df_json: str):
    df = pd.read_json(io.StringIO(df_json))

    if "Risk_Category" not in df.columns:
        return None, None, None, None, None, "Risk_Category column not found"

    drop_cols = [c for c in ["Risk_Category","Patient_ID"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Risk_Category"]
    mask = y.isin(["LOW","MEDIUM","HIGH","CRITICAL"])
    X, y = X[mask], y[mask]
    if len(y.unique()) < 2:
        return None, None, None, None, None, "Need at least 2 risk categories"

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    num_feats = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object"]).columns.tolist()

    transformers = []
    if num_feats: transformers.append(("num", StandardScaler(), num_feats))
    if cat_feats: transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_feats))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_proc = preprocessor.fit_transform(X)
    cat_out = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_feats).tolist() if cat_feats else []
    feature_names = num_feats + cat_out

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=15
    )
    model.fit(X_train, y_train)

    acc    = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), target_names=le.classes_)
    return model, preprocessor, le, feature_names, acc, report


# ============================================
# CACHED: SHAP
# ============================================
@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data(show_spinner=False)
def compute_shap(_explainer, X_proc_tuple: tuple, feature_names: list, class_idx: int):
    X_arr = np.array(X_proc_tuple).reshape(1, -1)
    shap_values = _explainer.shap_values(X_arr)
    patient_shap = (shap_values[class_idx][0] if isinstance(shap_values, list)
                    else shap_values[0, :, class_idx])
    impact_df = pd.DataFrame({"Feature": feature_names, "Impact": patient_shap})
    impact_df["Absolute_Impact"] = impact_df["Impact"].abs()
    return impact_df.sort_values("Absolute_Impact", ascending=False).head(5)

def predict_and_explain(model, preprocessor, label_encoder, feature_names, input_df):
    # Align input columns exactly to what the preprocessor expects
    drop = [c for c in ["Risk_Category","Patient_ID"] if c in input_df.columns]
    input_df = input_df.drop(columns=drop)

    # Add any missing engineered features
    input_df = add_engineered_features(input_df)

    # Filter and enforce the exact column order expected by the preprocessor
    expected = list(preprocessor.feature_names_in_)
    input_df = input_df[expected]

    X_proc    = preprocessor.transform(input_df)
    pred      = model.predict(X_proc)
    label     = label_encoder.inverse_transform(pred)[0].upper()
    explainer = get_explainer(model)
    X_tuple   = tuple(X_proc[0].tolist())
    impact_df = compute_shap(explainer, X_tuple, feature_names, int(pred[0]))
    
    return label, impact_df


# ============================================
# SHAP CHART
# ============================================
def render_shap_chart(impact_df, title):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    colors = ["#ff6b35" if v > 0 else "#00e5a0" for v in impact_df["Impact"]]
    ax.barh(impact_df["Feature"], impact_df["Impact"],
            color=colors, edgecolor="none", height=0.50)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0)
    ax.set_xlabel("Impact on Predicted Risk", fontweight="500",
                  color=FONT_COLOR, fontsize=10, fontfamily="monospace")
    ax.tick_params(colors=FONT_COLOR, labelsize=9)
    ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color=GRID_COLOR)
    ax.invert_yaxis()
    plt.tight_layout()
    st.markdown(f"#### {title}")
    st.pyplot(fig)
    plt.close()


# ============================================
# CLINICAL RECOMMENDATIONS
# ============================================
def render_recommendations(impact_df):
    top  = impact_df[impact_df["Impact"] > 0]["Feature"].tolist()
    recs = []
    if "HbA1c_percent" in top or "Metabolic_Risk_Score" in top:
        recs.append(("🩸","Endocrinology","High HbA1c. Initiate glycemic control protocol."))
    if any(f in top for f in ["MAP","Systolic_BP","Diastolic_BP"]):
        recs.append(("🫀","Cardiology","BP markers driving risk. Review antihypertensives."))
    if "Smoking_Status_Current Smoker" in top:
        recs.append(("🚭","Lifestyle","Refer to cessation program immediately."))
    if any(f in top for f in ["Physical_Activity_hours_per_week","Daily_Steps","Activity_Score"]):
        recs.append(("🏃","Physical Therapy","Prescribe graduated mobility plan."))
    if any(f in top for f in ["Sleep_Hours_per_night","Sleep_Quality"]):
        recs.append(("😴","Sleep Medicine","Poor sleep detected. Evaluate for disorders."))
    if "BMI_Age_Index" in top or "BMI" in top:
        recs.append(("⚖️","Nutrition","Elevated BMI. Refer to dietitian."))
    if not recs:
        recs.append(("✅","Monitor","No critical interventions required."))
    st.markdown("#### 🩺 Clinical Pathway")
    for icon, lbl, detail in recs:
        st.info(f"{icon} **{lbl}:** {detail}")
    return recs


# ============================================
# PDF REPORT
# ============================================
def make_pdf(title, items):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()
    els = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for item in items:
        els.append(Paragraph(item, styles["Normal"]))
        els.append(Spacer(1, 6))
    doc.build(els)
    buf.seek(0)
    return buf


# ============================================
# LOAD DEFAULT ARTIFACTS (if present on disk)
# ============================================
if not st.session_state["model_trained"]:
    try:
        if os.path.exists("models/rf_model.pkl"):
            st.session_state["model"]         = joblib.load("models/rf_model.pkl")
            st.session_state["preprocessor"]  = joblib.load("models/preprocessor.pkl")
            st.session_state["label_encoder"] = joblib.load("models/label_encoder.pkl")
            st.session_state["feature_names"] = joblib.load("models/feature_names.pkl")
            st.session_state["model_trained"] = True
    except Exception: pass
    try:
        default_csv = "data/Healthcare_Risk_Classification_Dataset_Balanced.csv"
        if os.path.exists(default_csv) and st.session_state["df"] is None:
            df_pre = pd.read_csv(default_csv)
            df_pre["Risk_Category"] = df_pre["Risk_Category"].str.upper()
            df_pre = add_engineered_features(df_pre)   # ensure MAP etc. exist
            st.session_state["df"] = df_pre
    except Exception: pass


# ============================================
# TABS
# ============================================
tab_dash, tab_db, tab_manual = st.tabs([
    "📊  DASHBOARD",
    "🗄️  PATIENT DB",
    "✍️  ASSESSMENT"
])


# ==========================================
# TAB 1 — EXECUTIVE DASHBOARD
# ==========================================
with tab_dash:
    st.markdown("<div class='section-title'>📂 Upload Patient Dataset</div>", unsafe_allow_html=True)
    st.caption("upload raw csv or excel — auto-cleaned, feature-engineered, model retrained")

    uploaded_file = st.file_uploader("drop file", type=["csv","xlsx"], label_visibility="collapsed")

    if uploaded_file:
        raw_bytes = uploaded_file.read()
        fhash     = file_hash(raw_bytes)

        if fhash != st.session_state.get("last_file_hash"):
            st.session_state["last_file_hash"] = fhash

            with st.spinner("cleaning & engineering features…"):
                cleaned_df, cleaning_log = auto_clean_and_engineer(raw_bytes)

            st.success(f"✅ {uploaded_file.name} — {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} cols")

            with st.expander("pipeline execution log", expanded=False):
                for icon, label, detail in cleaning_log:
                    pipeline_step(icon, label, detail)

            if "Risk_Category" in cleaned_df.columns:
                with st.spinner("training model…"):
                    result = train_model_cached(fhash, cleaned_df.to_json())

                if result[0] is not None:
                    model_new, prep_new, le_new, fn_new, acc, report = result
                    st.session_state.update({
                        "df": cleaned_df, "model": model_new,
                        "preprocessor": prep_new, "label_encoder": le_new,
                        "feature_names": fn_new, "model_trained": True
                    })
                    st.success(f"🤖 model trained — {acc*100:.1f}% accuracy")
                else:
                    st.error(f"training failed: {result[-1]}")
                    st.session_state["df"] = cleaned_df
            else:
                st.session_state["df"] = cleaned_df
                st.warning("no risk_category column — data cleaned, manual assessment available")
        else:
            st.info("same file — using cached results")

    st.markdown("<hr>", unsafe_allow_html=True)
    df = st.session_state.get("df")

    if df is None:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px;'>
                <div style='font-size:36px; opacity:0.12; margin-bottom:12px;'>📁</div>
                <div style='font-family:"Syne",sans-serif; font-size:15px;
                            font-weight:700; color:#252a3a;'>no dataset loaded</div>
                <div style='font-size:11px; margin-top:6px; color:#252a3a;
                            font-family:"DM Mono",monospace;'>upload a csv or excel file above</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-title'>📊 Population Health Overview</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records", f"{len(df):,}")
        if "Risk_Category" in df.columns:
            c2.metric("Critical", f"{len(df[df['Risk_Category']=='CRITICAL']):,}")
            c3.metric("High Risk", f"{len(df[df['Risk_Category']=='HIGH']):,}")
        c4.metric("Avg BMI", f"{df['BMI'].mean():.1f}" if "BMI" in df.columns else "—")
        c5.metric("Avg Age", f"{df['Age'].mean():.1f}" if "Age" in df.columns else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        if "Risk_Category" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, names="Risk_Category", title="Risk Distribution",
                             hole=0.50, color="Risk_Category", color_discrete_map=RISK_COLOR_MAP)
                fig.update_traces(textposition="inside", textinfo="percent+label",
                                  textfont_color="white", textfont_size=11)
                fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "Age" in df.columns and "BMI" in df.columns:
                    sz  = "Systolic_BP" if "Systolic_BP" in df.columns else None
                    hov = ["Cholesterol_mg_dL"] if "Cholesterol_mg_dL" in df.columns else []
                    fig2 = px.scatter(df.sample(min(500, len(df)), random_state=1),
                                      x="Age", y="BMI", color="Risk_Category", size=sz,
                                      hover_data=hov, title="Age vs BMI",
                                      color_discrete_map=RISK_COLOR_MAP, opacity=0.80)
                    fig2.update_layout(margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                    st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if "Metabolic_Risk_Score" in df.columns:
                    fig3 = px.box(df, x="Risk_Category", y="Metabolic_Risk_Score",
                                  color="Risk_Category", color_discrete_map=RISK_COLOR_MAP,
                                  title="Metabolic Risk Score")
                    fig3.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                    st.plotly_chart(fig3, use_container_width=True)
                elif "HbA1c_percent" in df.columns:
                    fig3 = px.histogram(df, x="HbA1c_percent", color="Risk_Category",
                                        barmode="overlay", color_discrete_map=RISK_COLOR_MAP,
                                        title="HbA1c Distribution", opacity=0.82)
                    fig3.update_layout(margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                    st.plotly_chart(fig3, use_container_width=True)

            with col4:
                if "Smoking_Status" in df.columns:
                    smoke = df.groupby(["Smoking_Status","Risk_Category"]).size().reset_index(name="Count")
                    fig4  = px.bar(smoke, x="Smoking_Status", y="Count",
                                   color="Risk_Category", barmode="group",
                                   color_discrete_map=RISK_COLOR_MAP, title="Risk by Smoking Status")
                    fig4.update_layout(margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                    st.plotly_chart(fig4, use_container_width=True)
                elif "Activity_Score" in df.columns:
                    fig4 = px.box(df, x="Risk_Category", y="Activity_Score",
                                  color="Risk_Category", color_discrete_map=RISK_COLOR_MAP,
                                  title="Activity Score")
                    fig4.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **plotly_layout())
                    st.plotly_chart(fig4, use_container_width=True)

        with st.expander("preview dataset — first 20 rows"):
            st.dataframe(df.head(20), use_container_width=True)

        if st.session_state.get("model_trained"):
            st.info("→ open patient db tab for per-patient diagnostics, or assessment tab for manual entry")


# ==========================================
# TAB 2 — PATIENT DATABASE
# ==========================================
with tab_db:
    df            = st.session_state.get("df")
    model         = st.session_state.get("model")
    preprocessor  = st.session_state.get("preprocessor")
    label_encoder = st.session_state.get("label_encoder")
    feature_names = st.session_state.get("feature_names")

    st.markdown("<div class='section-title'>🗄️ Search Existing Patient Records</div>", unsafe_allow_html=True)

    if df is None:
        st.warning("no dataset loaded — upload in the dashboard tab")
    elif not st.session_state.get("model_trained"):
        st.warning("model not trained — upload a labelled dataset in the dashboard tab")
    else:
        with st.expander("view active ehr data table"):
            st.dataframe(df.head(50), use_container_width=True)

        if "Patient_ID" in df.columns:
            selected_id = st.selectbox("select patient id:", df["Patient_ID"].tolist())

            if st.button("🧬  Run AI Diagnostic", use_container_width=True):
                patient_row = df[df["Patient_ID"] == selected_id].copy()
                actual_risk = (patient_row["Risk_Category"].values[0].upper()
                               if "Risk_Category" in patient_row.columns else "N/A")

                # Drop label + id, then re-apply engineered features to guarantee MAP etc. exist
                input_data = patient_row.drop(
                    columns=[c for c in ["Risk_Category","Patient_ID"] if c in patient_row.columns]
                )
                input_data = add_engineered_features(input_data)

                try:
                    with st.spinner("running prediction…"):
                        predicted_label, impact_df = predict_and_explain(
                            model, preprocessor, label_encoder, feature_names, input_data
                        )
                    st.markdown("---")
                    render_risk_badge(predicted_label)

                    colA, colB = st.columns(2)
                    with colA: st.info(f"📋 **historical diagnosis:** {actual_risk}")
                    with colB:
                        if actual_risk == predicted_label:
                            st.success("🤖 **ai agreement:** match confirmed")
                        else:
                            st.warning("⚠️ **ai agreement:** divergence detected")

                    st.markdown("---")
                    col_c, col_r = st.columns([1.2, 1])
                    with col_c:
                        render_shap_chart(impact_df, f"AI Logic — {selected_id}")
                    with col_r:
                        recs = render_recommendations(impact_df)

                    pdf_items = [
                        f"<b>Patient ID:</b> {selected_id}",
                        f"<b>AI Predicted Risk:</b> {predicted_label}",
                        f"<b>Historical Diagnosis:</b> {actual_risk}",
                        "<b>Top Risk Drivers:</b>",
                    ] + [f"• {r['Feature']}: {r['Impact']:.4f}" for _, r in impact_df.iterrows()]

                    buf = make_pdf(f"Patient Risk Report — {selected_id}", pdf_items)
                    st.download_button(
                        f"📥 download report — {selected_id}", buf,
                        file_name=f"Patient_{selected_id}_Report.pdf",
                        mime="application/pdf", key=f"db_dl_{selected_id}"
                    )
                except Exception as e:
                    st.error(f"prediction error: {e}")
        else:
            st.warning("no patient_id column found")


# ==========================================
# TAB 3 — CLINICAL ASSESSMENT
# ==========================================
with tab_manual:
    model         = st.session_state.get("model")
    preprocessor  = st.session_state.get("preprocessor")
    label_encoder = st.session_state.get("label_encoder")
    feature_names = st.session_state.get("feature_names")

    st.markdown("<div class='section-title'>✍️ New Patient Clinical Assessment</div>", unsafe_allow_html=True)

    if not st.session_state.get("model_trained"):
        st.warning("no trained model — upload a labelled dataset in the dashboard tab first")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**demographics & metabolic**")
            age         = st.slider("Age", 25, 85, 40)
            bmi         = st.number_input("BMI", 15.0, 45.0, 25.0)
            gender      = st.selectbox("Gender", ["Male","Female"])
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 350, 200)
            hba1c       = st.number_input("HbA1c (%)", 4.0, 15.0, 6.0)
        with col2:
            st.markdown("**cardiovascular & activity**")
            systolic_bp  = st.number_input("Systolic BP", 90, 200, 120)
            diastolic_bp = st.number_input("Diastolic BP", 60, 120, 80)
            heart_rate   = st.number_input("Heart Rate (bpm)", 50, 120, 70)
            steps        = st.number_input("Daily Steps", 0, 20000, 5000)
            alcohol      = st.slider("Alcohol units/week", 0, 20, 2)
        with col3:
            st.markdown("**lifestyle & family history**")
            smoking         = st.selectbox("Smoking Status", ["Non-Smoker","Former Smoker","Current Smoker"])
            activity        = st.slider("Activity hrs/week", 0.0, 15.0, 3.0)
            sleep           = st.slider("Sleep hrs/night", 3.0, 10.0, 7.0)
            family_diabetes = st.selectbox("Family Hx Diabetes", [0, 1])
            family_heart    = st.selectbox("Family Hx Heart Disease", [0, 1])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧬  Evaluate Patient", use_container_width=True):
            input_data = pd.DataFrame({
                "Age":[age], "Gender":[gender], "BMI":[bmi],
                "Systolic_BP":[systolic_bp], "Diastolic_BP":[diastolic_bp],
                "Cholesterol_mg_dL":[cholesterol], "HbA1c_percent":[hba1c],
                "Smoking_Status":[smoking], "Alcohol_Consumption_per_week":[alcohol],
                "Physical_Activity_hours_per_week":[activity], "Sleep_Hours_per_night":[sleep],
                "Avg_Heart_Rate":[heart_rate], "Daily_Steps":[steps],
                "Family_History_Diabetes":[family_diabetes],
                "Family_History_Heart_Disease":[family_heart]
            })
            # Add all engineered features via shared helper
            input_data = add_engineered_features(input_data)

            try:
                with st.spinner("running prediction…"):
                    predicted_label, impact_df = predict_and_explain(
                        model, preprocessor, label_encoder, feature_names, input_data
                    )
                st.markdown("---")
                render_risk_badge(predicted_label)

                col_c, col_r = st.columns([1.2, 1])
                with col_c:
                    render_shap_chart(impact_df, "AI Logic — New Patient")
                with col_r:
                    recs = render_recommendations(impact_df)

                pdf_items = [
                    "<b>Assessment:</b> Manual Entry",
                    f"<b>AI Risk:</b> {predicted_label}",
                    f"<b>Age:</b> {age}  <b>Gender:</b> {gender}  <b>BMI:</b> {bmi}  <b>HbA1c:</b> {hba1c}%",
                    f"<b>BP:</b> {systolic_bp}/{diastolic_bp} mmHg",
                    "<b>Top Risk Drivers:</b>",
                ] + [f"• {r['Feature']}: {r['Impact']:.4f}" for _, r in impact_df.iterrows()]

                buf = make_pdf("Patient Risk Report — Manual Assessment", pdf_items)
                st.download_button(
                    "📥 download assessment report", buf,
                    file_name="New_Patient_Assessment_Report.pdf",
                    mime="application/pdf", key="manual_dl"
                )
            except Exception as e:
                st.error(f"prediction error: {e}")

# ============================================
# LOGOUT
# ============================================
st.markdown("<br><hr>", unsafe_allow_html=True)
_, col_logout = st.columns([9, 1])
with col_logout:
    st.button("exit", on_click=lambda: st.session_state.update(logged_in=False))

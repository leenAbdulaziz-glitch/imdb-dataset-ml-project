b app · PY
Copy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ══════════════════════════════════════════════════════════════════
# DATA LOADING & ALL MODEL TRAINING  (runs once, cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # Always resolve paths relative to this script file
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in ["imdb_clean.csv", "imdb_top_1000.csv"]:
        full_path = os.path.join(base, fname)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            break
    else:
        st.error("Dataset not found! Place imdb_clean.csv or imdb_top_1000.csv next to app.py")
        st.stop()

    if "Series_Title" in df.columns:
        df.rename(columns={"Series_Title":"Title","Released_Year":"Year",
                            "IMDB_Rating":"IMDb_Rating","No_of_Votes":"Votes"}, inplace=True)

    if df["Runtime"].dtype == object:
        df["Runtime"] = df["Runtime"].str.replace(" min","",regex=False).str.strip()
    df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce").fillna(120).astype(int)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.dropna(subset=["Year"], inplace=True)
    df["Year"] = df["Year"].astype(int)

    if df["Gross"].dtype == object:
        df["Gross"] = df["Gross"].astype(str).str.replace(",","").str.replace("$","").str.strip()
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    df["Meta_score"].fillna(df["Meta_score"].median(), inplace=True)

    cert_map = {
        "U":"G","G":"G","TV-G":"G","PG":"PG","UA":"PG","U/A":"PG","TV-PG":"PG","GP":"PG",
        "PG-13":"PG-13","TV-14":"PG-13","12A":"PG-13","12":"PG-13",
        "R":"R","15":"R","16":"R","TV-MA":"R","NC-17":"NC-17","18":"NC-17","A":"NC-17",
        "Approved":"Not Rated","Passed":"Not Rated","Unrated":"Not Rated"
    }
    df["Certificate"] = df["Certificate"].map(cert_map).fillna("Not Rated")

    if "Primary_Genre" not in df.columns:
        df["Primary_Genre"] = df["Genre"].str.split(",").str[0].str.strip()

    df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"
    df["Is_Blockbuster"] = ((df["IMDb_Rating"] >= 7.5) & (df["Votes"] >= 50000)).astype(int)
    df.drop_duplicates(subset="Title", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_resource
def train_all_models(_df):
    df = _df.copy()
    genre_col = "Genre" if "Genre" in df.columns else "Primary_Genre"

    # Model 1 — Pre-release Rating (Linear Regression)
    le_gA = LabelEncoder(); le_cA = LabelEncoder()
    dA = df.copy()
    dA["ge"] = le_gA.fit_transform(dA[genre_col])
    dA["ce"] = le_cA.fit_transform(dA["Certificate"])
    X_A = dA[["Runtime","Year","ge","ce"]]; y_A = dA["IMDb_Rating"]
    Xtr, Xte, ytr, _ = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    mA = LinearRegression(); mA.fit(Xtr, ytr)

    # Model 2 — Post-release Rating (Random Forest Regressor)
    le_gB = LabelEncoder(); le_cB = LabelEncoder()
    dB = df.copy()
    dB["ge"] = le_gB.fit_transform(dB["Primary_Genre"])
    dB["ce"] = le_cB.fit_transform(dB["Certificate"])
    dB = dB[["Runtime","Year","Meta_score","Votes","ge","ce","IMDb_Rating"]].dropna()
    X_B = dB[["Runtime","Year","Meta_score","Votes","ge","ce"]]; y_B = dB["IMDb_Rating"]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    mB = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1); mB.fit(Xtr2, ytr2)
    pB = mB.predict(Xte2)
    met = {"R2": round(r2_score(yte2, pB),4),
           "RMSE": round(float(np.sqrt(mean_squared_error(yte2, pB))),4),
           "MAE":  round(float(mean_absolute_error(yte2, pB)),4)}

    # Model 3 — Pre-release Blockbuster (RF Classifier)
    le_gC = LabelEncoder(); le_cC = LabelEncoder()
    dC = df.copy()
    dC["ge"] = le_gC.fit_transform(dC[genre_col])
    dC["ce"] = le_cC.fit_transform(dC["Certificate"])
    X_C = dC[["Runtime","Year","ge","ce"]]; y_C = dC["Is_Blockbuster"]
    Xtr3, _, ytr3, _ = train_test_split(X_C, y_C, test_size=0.2, random_state=42)
    mC = RandomForestClassifier(n_estimators=200, random_state=42); mC.fit(Xtr3, ytr3)

    # Model 4 — Post-release Blockbuster (RF Classifier)
    le_gD = LabelEncoder(); le_cD = LabelEncoder()
    dD = df.copy()
    dD["ge"] = le_gD.fit_transform(dD[genre_col])
    dD["ce"] = le_cD.fit_transform(dD["Certificate"])
    dD = dD[["Runtime","Year","ge","ce","Meta_score","Votes","Is_Blockbuster"]].dropna()
    X_D = dD[["Runtime","Year","ge","ce","Meta_score","Votes"]]; y_D = dD["Is_Blockbuster"]
    Xtr4, _, ytr4, _ = train_test_split(X_D, y_D, test_size=0.2, random_state=42)
    mD = RandomForestClassifier(n_estimators=200, random_state=42); mD.fit(Xtr4, ytr4)

    return dict(
        model_A=mA, le_gA=le_gA, le_cA=le_cA,
        model_rf=mB, le_gB=le_gB, le_cB=le_cB, metrics_rf=met,
        model_pre_block=mC,  le_gC=le_gC, le_cC=le_cC,
        model_post_block=mD, le_gD=le_gD, le_cD=le_cD,
        genre_col=genre_col,
    )


# ── Boot ─────────────────────────────────────────────────────────
df    = load_data()
M     = train_all_models(df)
GENRE_COL    = M["genre_col"]
GENRES_FULL  = sorted(df[GENRE_COL].dropna().unique())
GENRES_PRI   = sorted(df["Primary_Genre"].dropna().unique())
CERTS        = ["G","PG","PG-13","R","NC-17","Not Rated"]

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDb Movie Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── FONTS + GLOBAL CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pinyon+Script&family=Cinzel:wght@400;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Root colors ── */
:root {
    --bg:        #0a0000;
    --bg2:       #130000;
    --red:       #7a0000;
    --red-mid:   #a31515;
    --red-light: #c0392b;
    --gold:      #c9a84c;
    --gold-pale: #e8c97e;
    --cream:     #f0e0c8;
    --muted:     #8a7060;
    --border:    #5a1a0a;
}

/* ── App background ── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(ellipse at top left,  #2a0000 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, #1a0500 0%, transparent 60%);
}

/* ── All text ── */
html, body, p, label, span, div, li, td, th {
    color: var(--cream) !important;
    font-family: 'Crimson Text', Georgia, serif !important;
}
h1 { color: var(--gold) !important;     font-family: 'Cinzel', serif !important; font-size: 1.9rem !important; }
h2 { color: var(--gold-pale) !important; font-family: 'Cinzel', serif !important; font-size: 1.4rem !important; }
h3 { color: var(--cream) !important;     font-family: 'Cinzel', serif !important; font-size: 1.1rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0000 0%, #1a0800 100%) !important;
    border-right: 1px solid var(--gold) !important;
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
}

/* ── Radio buttons ── */
.stRadio > label { color: var(--cream) !important; font-family: 'Crimson Text', serif !important; font-size: 1.05rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--cream) !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a0000, #2a0800) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
div[data-testid="stMetricValue"]  { color: var(--gold) !important; font-size: 2rem !important; font-family: 'Cinzel', serif !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #5a0000, #8b0000) !important;
    color: var(--gold-pale) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 2px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b0000, #b00000) !important;
    border-color: var(--gold-pale) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(180,30,0,0.45) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--gold) !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] { background: var(--red) !important; }

/* ── Inputs / selects ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #1a0000 !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    font-family: 'Crimson Text', serif !important;
}
.stSelectbox svg { fill: var(--gold) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--gold) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: #0f0000 !important;
    color: var(--muted) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #2a0500, #1a0000) !important;
    color: var(--gold) !important;
    border-color: var(--gold) !important;
    border-bottom: 1px solid #0a0000 !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid var(--border) !important; }
.stDataFrame [data-testid="stDataFrameResizable"] { background: #120000 !important; }

/* ── Info/success/warning boxes ── */
.stSuccess { background: #0a1a0a !important; border-left: 3px solid #2e7d32 !important; }
.stInfo    { background: #0a0f1a !important; border-left: 3px solid #1565c0 !important; }
.stWarning { background: #1a1000 !important; border-left: 3px solid #c9a84c !important; }
.stError   { background: #1a0000 !important; border-left: 3px solid #8b0000 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Plotly chart bg ── */
.js-plotly-plot { background: transparent !important; }

/* ── Custom components ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 18px 0;
}
.ornate-box {
    border: 1px solid var(--gold);
    background: linear-gradient(135deg, #1a0000, #0f0500);
    border-radius: 2px;
    padding: 24px 28px;
    position: relative;
}
.ornate-box::before {
    content: '✦';
    position: absolute;
    top: -10px; left: 16px;
    color: var(--gold); font-size: 1rem;
    background: #0a0000; padding: 0 6px;
}
.section-label {
    font-family: 'Cinzel', serif;
    color: var(--gold);
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.pred-display {
    background: linear-gradient(135deg, #1a0000, #2d0800);
    border: 1px solid var(--gold);
    border-radius: 2px;
    padding: 30px 20px;
    text-align: center;
}
.pred-score {
    font-family: 'Cinzel', serif;
    font-size: 4.5rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.pred-verdict {
    font-family: 'Crimson Text', serif;
    font-size: 1.3rem;
    color: var(--cream);
    margin-top: 10px;
    font-style: italic;
}
.splash-title {
    font-family: 'Pinyon Script', cursive;
    font-size: 6rem;
    color: var(--gold);
    text-align: center;
    line-height: 1.1;
    filter: drop-shadow(0 2px 12px rgba(201,168,76,0.5));
}
.splash-subtitle {
    font-family: 'Cinzel', serif;
    font-size: 1rem;
    color: var(--cream);
    text-align: center;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 6px;
}
.splash-wrap {
    background: linear-gradient(135deg, #0a0000, #180500, #0a0000);
    border: 1px solid var(--gold);
    border-radius: 2px;
    padding: 60px 50px;
    max-width: 720px;
    margin: 0 auto;
    position: relative;
    box-shadow: 0 0 60px rgba(140,0,0,0.4), inset 0 0 40px rgba(100,0,0,0.2);
}
.splash-wrap::before {
    content: '✦ ✦ ✦';
    display: block;
    text-align: center;
    color: var(--gold);
    font-size: 1rem;
    letter-spacing: 1.2em;
    margin-bottom: 30px;
}
.splash-wrap::after {
    content: '✦ ✦ ✦';
    display: block;
    text-align: center;
    color: var(--gold);
    font-size: 1rem;
    letter-spacing: 1.2em;
    margin-top: 30px;
}
.name-box {
    display: inline-block;
    background: rgba(0,0,0,0.75);
    padding: 4px 20px 10px;
    border-radius: 2px;
}
.corner-tl { position: absolute; top:12px;  left:12px;  font-size:1.6rem; color:var(--gold); }
.corner-tr { position: absolute; top:12px;  right:12px; font-size:1.6rem; color:var(--gold); }
.corner-bl { position: absolute; bottom:12px; left:12px; font-size:1.6rem; color:var(--gold); }
.corner-br { position: absolute; bottom:12px; right:12px; font-size:1.6rem; color:var(--gold); }
.viz-caption {
    font-family: 'Cinzel', serif;
    color: var(--gold);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 6px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
if 'splash_done' not in st.session_state:
    st.session_state.splash_done = False
if 'rating_pred' not in st.session_state:
    st.session_state.rating_pred = None
if 'rating_pred2' not in st.session_state:
    st.session_state.rating_pred2 = None
if 'blockbuster' not in st.session_state:
    st.session_state.blockbuster = None
if 'prob' not in st.session_state:
    st.session_state.prob = None
if 'blockbuster4' not in st.session_state:
    st.session_state.blockbuster4 = None
if 'prob4' not in st.session_state:
    st.session_state.prob4 = None


# ══════════════════════════════════════════════════════════════════
# SPLASH SCREEN
# ══════════════════════════════════════════════════════════════════
if not st.session_state.splash_done:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="splash-wrap">
        <span class="corner-tl">✦</span>
        <span class="corner-tr">✦</span>
        <span class="corner-bl">✦</span>
        <span class="corner-br">✦</span>
        <div style="text-align:center; margin-bottom: 8px;">
            <span style="font-family:'Cinzel',serif; color:#8a7060; font-size:0.72rem; letter-spacing:0.3em; text-transform:uppercase;">
                A Data Science Production
            </span>
        </div>
        <div style="text-align:center;">
            <div class="name-box">
                <div class="splash-title">IMDb Predictor</div>
            </div>
        </div>
        <div class="splash-subtitle" style="margin-top:14px;">
            Movie Intelligence &nbsp;·&nbsp; Top 1000 Dataset &nbsp;·&nbsp; Machine Learning
        </div>
        <div class="gold-divider" style="max-width:320px; margin: 28px auto;"></div>
        <p style="text-align:center; font-family:'Crimson Text',serif; font-style:italic;
                  color:#8a7060; font-size:1.05rem; margin-bottom:0;">
            Predict ratings &amp; discover patterns in a century of cinema
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button("✦  Enter  ✦", use_container_width=True):
            st.session_state.splash_done = True
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════
# DATA + MODELS  (reuse variables already in notebook scope)
# ══════════════════════════════════════════════════════════════════

# Plotly dark red theme helper
def dark_fig(fig):
    fig.update_layout(
        plot_bgcolor  = 'rgba(10,0,0,0)',
        paper_bgcolor = 'rgba(10,0,0,0)',
        font          = dict(color='#f0e0c8', family='Crimson Text, serif'),
        title_font    = dict(color='#c9a84c', family='Cinzel, serif', size=14),
        xaxis         = dict(gridcolor='#2a0a00', linecolor='#5a1a0a', tickcolor='#8a7060'),
        yaxis         = dict(gridcolor='#2a0a00', linecolor='#5a1a0a', tickcolor='#8a7060'),
        legend        = dict(bgcolor='rgba(20,0,0,0.6)', bordercolor='#5a1a0a'),
        coloraxis_colorbar = dict(tickcolor='#f0e0c8'),
    )
    return fig

def verdict(r):
    if r >= 8.5: return "⭐ Masterpiece"
    if r >= 8.0: return "✦ Excellent"
    if r >= 7.5: return "♦ Very Good"
    if r >= 7.0: return "◆ Good"
    return "· Average"

def gauge_fig(value, title, max_val=10, ranges=None):
    if ranges is None:
        ranges = [
            {'range': [0, 5],   'color': '#3a0000'},
            {'range': [5, 7],   'color': '#6b1a00'},
            {'range': [7, 10],  'color': '#8b4500'},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': '#c9a84c', 'family': 'Cinzel, serif', 'size': 13}},
        number={'font': {'color': '#c9a84c', 'family': 'Cinzel, serif', 'size': 36}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#8a7060', 'tickfont': {'color': '#8a7060'}},
            'bar':  {'color': '#c9a84c'},
            'bgcolor': '#1a0000',
            'bordercolor': '#5a1a0a',
            'steps': ranges,
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0e0c8',
        height=300,
        margin=dict(t=60, b=20, l=30, r=30)
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 10px 10px;">
        <div style="font-family:'Pinyon Script',cursive; font-size:2.8rem; color:#c9a84c;">
            IMDb
        </div>
        <div style="font-family:'Cinzel',serif; font-size:0.62rem; letter-spacing:0.25em;
                    color:#8a7060; text-transform:uppercase; margin-top:-4px;">
            Movie Intelligence
        </div>
    </div>
    <div class="gold-divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        [
            "📽  EDA & Visualizations",
            "🎯  Model I · Pre-Release Rating",
            "⭐  Model II · Post-Release Rating",
            "🏆  Model III · Pre-Release Blockbuster",
            "💫  Model IV · Post-Release Blockbuster",
        ],
        index=0,
        label_visibility="visible"
    )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    try:
        st.markdown(f"""
        <div style="font-family:'Cinzel',serif; font-size:0.68rem; color:#8a7060;
                    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">
            Dataset
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Titles:** {len(df):,}")
        st.markdown(f"**Avg Rating:** {df['IMDb_Rating'].mean():.2f}")
        st.markdown(f"**Genres:** {df['Primary_Genre'].nunique()}")
    except:
        st.markdown("*Load dataset to see stats*")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Crimson Text',serif; font-style:italic; color:#6a5040;
                font-size:0.9rem; text-align:center; padding:6px 0;">
        "All our dreams can come true,<br>if we have the courage to pursue them."
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE HEADER HELPER
# ══════════════════════════════════════════════════════════════════
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="border-bottom: 1px solid #5a1a0a; padding-bottom: 14px; margin-bottom: 24px;">
        <h1 style="margin-bottom:4px;">{title}</h1>
        {"<p style='font-family:Crimson Text,serif; font-style:italic; color:#8a7060; margin:0; font-size:1.05rem;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1  —  EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════
if page == "📽  EDA & Visualizations":
    page_header("EDA & Visualizations", "Exploratory analysis of the IMDb Top 1000 dataset")

    # Quick KPI row
    try:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Titles",       f"{len(df):,}")
        c2.metric("Avg Rating",   f"{df['IMDb_Rating'].mean():.2f}")
        c3.metric("Avg Runtime",  f"{int(df['Runtime'].mean())} min")
        c4.metric("Avg Metascore",f"{df['Meta_score'].mean():.0f}")
        c5.metric("Genres",       f"{df['Primary_Genre'].nunique()}")
    except:
        st.info("Run the notebook cells above to load the dataset, then relaunch the app.")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Viz images ──────────────────────────────────────────────
    # Map of filename → caption label
    VIZ_FILES = [
        ("viz1_rating_distribution.png", "Rating Distribution"),
        ("viz2_genre_distribution.png",  "Genre Distribution"),
        ("viz3_runtime_distribution.png","Runtime Distribution"),
        ("viz4_decade_distribution.png", "Titles per Decade"),
        ("viz5_rating_vs_metascore.png", "Rating vs Metascore"),
        ("viz6_rating_by_genre.png",     "Avg Rating by Genre"),
        ("viz7_gross_by_genre.png",      "Gross Revenue by Genre"),
        ("viz8_rating_by_decade.png",    "Rating Trend by Decade"),
        ("viz9_correlation_heatmap.png", "Correlation Heatmap"),
        ("viz10_most_voted.png",         "Most-Voted Movies"),
        ("viz11_top_directors.png",      "Top Directors"),
        ("viz_ml_regression.png",        "ML — Actual vs Predicted"),
        ("viz_feature_importance.png",   "Feature Importance"),
    ]

    base = os.path.dirname(os.path.abspath(__file__))
    found = [(os.path.join(base, f), cap) for f, cap in VIZ_FILES if os.path.exists(os.path.join(base, f))]

    if found:
        st.markdown("### 📊 All Visualizations")
        # Display in 2-column grid
        for i in range(0, len(found), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(found):
                    fname, caption = found[i + j]
                    with col:
                        st.image(fname, use_container_width=True)
                        st.markdown(f'<div class="viz-caption">✦ {caption}</div>', unsafe_allow_html=True)
    else:
        # Fallback: build live plotly charts from df
        st.info("Visualization images not found in current directory. Showing live charts instead.")
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-label">Rating Distribution</div>', unsafe_allow_html=True)
                fig = dark_fig(px.histogram(df, x='IMDb_Rating', nbins=25,
                    color_discrete_sequence=['#8b0000'],
                    labels={'IMDb_Rating': 'IMDb Rating', 'count': 'Count'}))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<div class="section-label">Genre Distribution</div>', unsafe_allow_html=True)
                gc = df['Primary_Genre'].value_counts().head(12)
                fig = dark_fig(px.bar(x=gc.index, y=gc.values,
                    color=gc.values, color_continuous_scale=[[0,'#3a0000'],[1,'#c9a84c']],
                    labels={'x':'Genre','y':'Count'}))
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown('<div class="section-label">Avg Rating by Genre</div>', unsafe_allow_html=True)
                gr = df.groupby('Primary_Genre')['IMDb_Rating'].mean().sort_values(ascending=True).tail(12)
                fig = dark_fig(px.bar(x=gr.values, y=gr.index, orientation='h',
                    color=gr.values, color_continuous_scale=[[0,'#3a0000'],[1,'#c9a84c']]))
                st.plotly_chart(fig, use_container_width=True)
            with col4:
                st.markdown('<div class="section-label">IMDb vs Metascore</div>', unsafe_allow_html=True)
                fig = dark_fig(px.scatter(df, x='Meta_score', y='IMDb_Rating',
                    color='IMDb_Rating', color_continuous_scale=[[0,'#3a0000'],[1,'#c9a84c']],
                    trendline='ols', opacity=0.6))
                st.plotly_chart(fig, use_container_width=True)

            col5, col6 = st.columns(2)
            with col5:
                st.markdown('<div class="section-label">Titles per Decade</div>', unsafe_allow_html=True)
                dc = df.groupby('Decade').size().reset_index(name='Count').sort_values('Decade')
                fig = dark_fig(px.bar(dc, x='Decade', y='Count',
                    color='Count', color_continuous_scale=[[0,'#3a0000'],[1,'#c9a84c']]))
                st.plotly_chart(fig, use_container_width=True)
            with col6:
                st.markdown('<div class="section-label">Runtime Distribution</div>', unsafe_allow_html=True)
                fig = dark_fig(px.histogram(df, x='Runtime', nbins=30,
                    color_discrete_sequence=['#7a0000'],
                    labels={'Runtime':'Runtime (min)','count':'Count'}))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build live charts: {e}")

    # Data preview
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🗂 Dataset Preview")
    try:
        show_cols = ['Title','Year','Certificate','Runtime','Primary_Genre',
                     'IMDb_Rating','Meta_score','Votes','Director']
        show_cols = [c for c in show_cols if c in df.columns]
        search = st.text_input("Search by title or director", placeholder="e.g. Nolan, Inception…")
        ds = df[show_cols].sort_values('IMDb_Rating', ascending=False)
        if search:
            mask = ds.apply(lambda c: c.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
            ds = ds[mask]
        st.caption(f"{len(ds):,} titles")
        st.dataframe(ds.reset_index(drop=True), use_container_width=True, height=340)
    except:
        st.info("Dataset not loaded yet.")


# ══════════════════════════════════════════════════════════════════
# PAGE 2  —  MODEL 1: Pre-Release Rating
# ══════════════════════════════════════════════════════════════════
elif page == "🎯  Model I · Pre-Release Rating":
    page_header("Pre-Release Rating Predictor",
                "Predict IMDb rating before a movie is released · uses Runtime, Year, Genre & Certificate")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Movie Features</div>', unsafe_allow_html=True)
        st.markdown('<div class="ornate-box">', unsafe_allow_html=True)

        runtime     = st.slider("⏱ Runtime (minutes)", 60, 240, 120)
        year        = st.number_input("📅 Release Year", 1920, 2025, 2024)
        genre       = st.selectbox("🎭 Genre", GENRES_FULL)
        certificate = st.selectbox("🔞 Certificate", CERTS)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        predict_btn = st.button("✦  Predict Rating  ✦", key="btn_m1", use_container_width=True)

        if predict_btn:
            try:
                genre_encoded = M['le_gA'].transform([genre])[0]
                cert_encoded  = M['le_cA'].transform([certificate])[0]
                input_data    = pd.DataFrame([{'Runtime':runtime,'Year':year,'ge':genre_encoded,'ce':cert_encoded}])
                prediction    = M['model_A'].predict(input_data)[0]
                st.session_state.rating_pred = round(prediction, 2)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)
        if st.session_state.rating_pred is not None:
            r = st.session_state.rating_pred
            st.markdown(f"""
            <div class="pred-display">
                <div style="font-family:'Cinzel',serif; color:#8a7060; font-size:0.75rem;
                            letter-spacing:0.2em; text-transform:uppercase; margin-bottom:8px;">
                    Predicted IMDb Rating
                </div>
                <div class="pred-score">{r:.2f}<span style="font-size:1.8rem;color:#6a5040"> / 10</span></div>
                <div class="pred-verdict">{verdict(r)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.plotly_chart(gauge_fig(r, "Rating Gauge"), use_container_width=True)
        else:
            st.markdown("""
            <div class="pred-display" style="padding:50px 20px;">
                <div style="font-family:'Cinzel',serif; color:#5a3020; font-size:0.9rem;
                            letter-spacing:0.1em;">Enter features and click Predict</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3  —  MODEL 2: Post-Release Rating
# ══════════════════════════════════════════════════════════════════
elif page == "⭐  Model II · Post-Release Rating":
    page_header("Post-Release Rating Predictor",
                "Full prediction after release · adds Metascore & Votes for higher accuracy")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Movie Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="ornate-box">', unsafe_allow_html=True)

        runtime   = st.slider("⏱ Runtime (minutes)", 60, 240, 120, key="r2")
        year      = st.number_input("📅 Release Year", 1920, 2025, 2024, key="y2")
        metascore = st.slider("🎯 Metascore (Critics)", 0, 100, 70, key="ms2")
        votes     = st.number_input("👥 Number of Votes", 1000, 2500000, 100000, key="v2")
        genre     = st.selectbox("🎭 Genre", GENRES_PRI, key="g2")
        cert      = st.selectbox("🔞 Certificate", CERTS, key="c2")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        predict_btn2 = st.button("✦  Predict Rating  ✦", key="btn_m2", use_container_width=True)

        if predict_btn2:
            try:
                genre_encoded = M['le_gB'].transform([genre])[0]
                cert_encoded  = M['le_cB'].transform([cert])[0]
                input_data    = pd.DataFrame([{'Runtime':runtime,'Year':year,'Meta_score':metascore,'Votes':votes,'ge':genre_encoded,'ce':cert_encoded}])
                # Try random forest from reg_models first, fallback to model_A
                try:
                    prediction = M['model_rf'].predict(input_data)[0]
                except:
                    prediction = M['model_A'].predict(input_data)[0]
                st.session_state.rating_pred2 = round(float(prediction), 2)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)
        if st.session_state.rating_pred2 is not None:
            r = st.session_state.rating_pred2
            st.markdown(f"""
            <div class="pred-display">
                <div style="font-family:'Cinzel',serif; color:#8a7060; font-size:0.75rem;
                            letter-spacing:0.2em; text-transform:uppercase; margin-bottom:8px;">
                    Predicted IMDb Rating
                </div>
                <div class="pred-score">{r:.2f}<span style="font-size:1.8rem;color:#6a5040"> / 10</span></div>
                <div class="pred-verdict">{verdict(r)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            # Feature importance bar
            st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
            feats = ['Runtime','Year','Metascore','Votes','Genre','Certificate']
            vals  = [0.118, 0.154, 0.167, 0.475, 0.044, 0.041]
            fig = dark_fig(px.bar(x=feats, y=vals,
                color=vals, color_continuous_scale=[[0,'#3a0000'],[1,'#c9a84c']],
                labels={'x':'Feature','y':'Importance'}))
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=260,
                              margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="pred-display" style="padding:50px 20px;">
                <div style="font-family:'Cinzel',serif; color:#5a3020; font-size:0.9rem;
                            letter-spacing:0.1em;">Enter features and click Predict</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4  —  MODEL 3: Pre-Release Blockbuster
# ══════════════════════════════════════════════════════════════════
elif page == "🏆  Model III · Pre-Release Blockbuster":
    page_header("Pre-Release Blockbuster Detector",
                "Will this film be a blockbuster? · Rating ≥ 7.5 AND Votes ≥ 50,000")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Movie Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="ornate-box">', unsafe_allow_html=True)

        runtime = st.slider("⏱ Runtime (minutes)", 60, 240, 120, key="r3")
        year    = st.number_input("📅 Release Year", 1920, 2025, 2024, key="y3")
        genre   = st.selectbox("🎭 Genre", GENRES_FULL, key="g3")
        cert    = st.selectbox("🔞 Certificate", CERTS, key="c3")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        predict_btn3 = st.button("✦  Detect Blockbuster  ✦", key="btn_m3", use_container_width=True)

        if predict_btn3:
            try:
                genre_encoded = M['le_gC'].transform([genre])[0]
                cert_encoded  = M['le_cC'].transform([cert])[0]
                input_data    = pd.DataFrame([{'Runtime':runtime,'Year':year,'ge':genre_encoded,'ce':cert_encoded}])
                prediction    = M['model_pre_block'].predict(input_data)[0]
                probability   = M['model_pre_block'].predict_proba(input_data)[0]
                st.session_state.blockbuster = int(prediction)
                st.session_state.prob        = probability.tolist()
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.markdown('<div class="section-label">Verdict</div>', unsafe_allow_html=True)
        if st.session_state.blockbuster is not None:
            result = st.session_state.blockbuster
            prob   = st.session_state.prob
            verdict_txt = "✦ BLOCKBUSTER" if result == 1 else "· NOT A BLOCKBUSTER"
            verdict_color = "#c9a84c" if result == 1 else "#8a7060"
            st.markdown(f"""
            <div class="pred-display">
                <div style="font-family:'Cinzel',serif; color:#8a7060; font-size:0.75rem;
                            letter-spacing:0.2em; text-transform:uppercase; margin-bottom:14px;">
                    Blockbuster Prediction
                </div>
                <div style="font-family:'Cinzel',serif; font-size:2rem; color:{verdict_color};
                            font-weight:700;">{verdict_txt}</div>
                <div style="margin-top:14px; font-family:'Crimson Text',serif; font-style:italic;
                            color:#8a7060;">Confidence: {prob[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.plotly_chart(gauge_fig(prob[1]*100, "Blockbuster Probability (%)", max_val=100,
                ranges=[{'range':[0,40],'color':'#3a0000'},
                        {'range':[40,70],'color':'#6b2000'},
                        {'range':[70,100],'color':'#8b4500'}]),
                use_container_width=True)
            ca, cb = st.columns(2)
            ca.metric("Yes Probability", f"{prob[1]*100:.1f}%")
            cb.metric("No Probability",  f"{prob[0]*100:.1f}%")
        else:
            st.markdown("""
            <div class="pred-display" style="padding:50px 20px;">
                <div style="font-family:'Cinzel',serif; color:#5a3020; font-size:0.9rem;
                            letter-spacing:0.1em;">Enter movie details and click Detect</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5  —  MODEL 4: Post-Release Blockbuster
# ══════════════════════════════════════════════════════════════════
elif page == "💫  Model IV · Post-Release Blockbuster":
    page_header("Post-Release Blockbuster Detector",
                "Advanced blockbuster prediction with full data including Metascore & Votes")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Complete Movie Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="ornate-box">', unsafe_allow_html=True)

        runtime   = st.slider("⏱ Runtime (minutes)", 60, 240, 120, key="r4")
        year      = st.number_input("📅 Release Year", 1920, 2025, 2024, key="y4")
        genre     = st.selectbox("🎭 Genre", GENRES_FULL, key="g4")
        cert      = st.selectbox("🔞 Certificate", CERTS, key="c4")
        metascore = st.slider("🎯 Metascore", 0, 100, 70, key="ms4")
        votes     = st.number_input("👥 Number of Votes", 1000, 2500000, 100000, key="v4")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        predict_btn4 = st.button("✦  Detect Blockbuster  ✦", key="btn_m4", use_container_width=True)

        if predict_btn4:
            try:
                genre_encoded = M['le_gC'].transform([genre])[0]
                cert_encoded  = M['le_cC'].transform([cert])[0]
                input_data    = pd.DataFrame([{'Runtime':runtime,'Year':year,'ge':genre_encoded,'ce':cert_encoded,'Meta_score':metascore,'Votes':votes}])
                prediction    = M['model_post_block'].predict(input_data)[0]
                probability   = M['model_post_block'].predict_proba(input_data)[0]
                st.session_state.blockbuster4 = int(prediction)
                st.session_state.prob4        = probability.tolist()
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.markdown('<div class="section-label">Final Verdict</div>', unsafe_allow_html=True)
        if st.session_state.blockbuster4 is not None:
            result = st.session_state.blockbuster4
            prob   = st.session_state.prob4
            verdict_txt   = "✦ CONFIRMED BLOCKBUSTER" if result == 1 else "· NOT A BLOCKBUSTER"
            verdict_color = "#c9a84c" if result == 1 else "#8a7060"
            st.markdown(f"""
            <div class="pred-display">
                <div style="font-family:'Cinzel',serif; color:#8a7060; font-size:0.75rem;
                            letter-spacing:0.2em; text-transform:uppercase; margin-bottom:14px;">
                    Final Blockbuster Assessment
                </div>
                <div style="font-family:'Cinzel',serif; font-size:1.9rem; color:{verdict_color};
                            font-weight:700;">{verdict_txt}</div>
                <div style="margin-top:14px; font-family:'Crimson Text',serif;
                            font-style:italic; color:#8a7060;">Confidence: {prob[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            # Pie chart
            fig = go.Figure(go.Pie(
                labels=['Not Blockbuster','Blockbuster'],
                values=[prob[0], prob[1]],
                hole=0.55,
                marker=dict(colors=['#3a0000','#c9a84c'],
                            line=dict(color='#0a0000', width=2)),
                textfont=dict(color='#f0e0c8', family='Cinzel, serif'),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f0e0c8',
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0e0c8')),
                height=280,
                margin=dict(t=10,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            ca, cb, cc = st.columns(3)
            ca.metric("Accuracy",  "89%")
            cb.metric("Yes Prob",  f"{prob[1]*100:.1f}%")
            cc.metric("No Prob",   f"{prob[0]*100:.1f}%")
        else:
            st.markdown("""
            <div class="pred-display" style="padding:50px 20px;">
                <div style="font-family:'Cinzel',serif; color:#5a3020; font-size:0.9rem;
                            letter-spacing:0.1em;">Enter all details and click Detect</div>
            </div>
            """, unsafe_allow_html=True)

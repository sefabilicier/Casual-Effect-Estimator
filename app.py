import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any
import sys
import os
import base64
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DataConfig, ModelConfig, UIConfig
from data.generator import CausalDataGenerator
from data.loaders import RealDataLoader
from models.traditional import (
    LinearRegressionEstimator, 
    IPTWEstimator,
    StratificationEstimator
)
from models.ml_based import (
    CausalForest,
    SLearner,
    TLearner,
    XLearner
)
from models.robust import (
    DoublyRobustEstimator,
    AIPWEstimator
)
from models.diagnostic import CausalDiagnostics
from evaluation.metrics import CausalMetrics
from evaluation.visualization import CausalVisualizer
from evaluation.benchmark import CausalBenchmark

st.set_page_config(
    page_title="Causal Effect Estimator",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
<style>
    /* Hide all Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    .stApp > header {display: none;}
    
    /* Remove all default padding and set max-width */
    .stApp {
        max-width: 1000px !important;
        padding: 0 !important;
        margin: 0 auto !important;
    }
    
    .stApp > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[class*="stAppViewBlockContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main content container - perfectly centered */
    .main {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 2rem 1.5rem !important;
    }
    
    .block-container {
        max-width: 900px !important;
        padding: 2rem 1.5rem !important;
        margin: 0 auto !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Headers - clean, bold, minimal */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        margin: 1.75rem 0 1rem 0;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 0.75rem;
    }
    
    .subsection-title {
        font-size: 1rem;
        font-weight: 600;
        color: #333333;
        margin: 1.25rem 0 0.75rem 0;
    }
    
    /* Input styling - clean, minimal borders */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        color: #000000 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.15s ease !important;
    }
    
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover,
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #000000 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 1px #000000 !important;
        outline: none !important;
    }
    
    /* Button styling - pure black and white */
    .stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        border: 1px solid #000000 !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f5f5f5 !important;
        border-color: #000000 !important;
    }
    
    /* Metrics grid - clean cards */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #000000;
        transform: translateY(-1px);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        line-height: 1;
    }
    
    .metric-unit {
        font-size: 0.85rem;
        color: #999999;
        margin-left: 2px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        background: #f5f5f5;
        color: #000000;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #e5e5e5;
        margin: 0.5rem 0;
    }
    
    .status-badge-success {
        background: #f5f5f5;
        border-left: 3px solid #22c55e;
    }
    
    .status-badge-warning {
        background: #fef9e7;
        border-left: 3px solid #f59e0b;
    }
    
    .status-badge-info {
        background: #f0f7ff;
        border-left: 3px solid #3b82f6;
    }
    
    /* Insight box - minimal, black accent border */
    .insight-box {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        border-left: 3px solid #000000;
    }
    
    .insight-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #000000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Info message */
    .info-message {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        color: #000000;
        font-size: 0.9rem;
        margin: 1rem 0;
        border-left: 3px solid #000000;
    }
    
    /* Tab styling - centered, clean, no colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        justify-content: center;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #666666 !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        transition: all 0.15s ease !important;
        font-size: 0.95rem !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000 !important;
        background: #f5f5f5 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2px solid #000000 !important;
        background: transparent !important;
    }
    
    /* Radio buttons - pill style */
    .stRadio > div {
        gap: 0.75rem !important;
        flex-wrap: wrap !important;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 0.75rem !important;
    }
    
    .stRadio [role="radio"] {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 20px !important;
        padding: 0.4rem 1.25rem !important;
        cursor: pointer !important;
        font-size: 0.9rem !important;
    }
    
    .stRadio [role="radio"][aria-checked="true"] {
        background: #000000 !important;
        border-color: #000000 !important;
        color: white !important;
    }
    
    .stRadio [role="radio"] > div:first-child {
        display: none !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        margin: 0.5rem 0 !important;
    }
    
    .stCheckbox > div > div > div {
        border-color: #e5e5e5 !important;
    }
    
    .stCheckbox [aria-checked="true"] > div > div > div {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #e5e5e5 !important;
        height: 4px !important;
    }
    
    .stSlider [role="slider"] {
        background: #000000 !important;
        border: 2px solid white !important;
        width: 16px !important;
        height: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin-top: -6px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #000000 !important;
        background: #fafafa !important;
        border-radius: 6px !important;
        border: 1px solid #e5e5e5 !important;
        padding: 0.75rem 1rem !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e5e5e5 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 1.5rem !important;
    }
    
    /* Data table */
    .dataframe-container {
        background: white;
        border-radius: 6px;
        border: 1px solid #e5e5e5;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border: none !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        font-size: 0.9rem !important;
    }
    
    /* Plotly charts - minimal, no backgrounds */
    .js-plotly-plot {
        border-radius: 6px;
        overflow: hidden;
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: #e5e5e5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999999;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e5e5;
    }
    
    /* Hide default labels */
    label[data-testid="stWidgetLabel"] {
        display: none;
    }
    
    /* Metric columns layout */
    .metric-columns {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #e5e5e5 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #000000 !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metrics-row {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .metric-columns {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem !important;
            font-size: 0.85rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def format_number(num):
    """Format large numbers with commas"""
    if pd.isna(num) or num is None:
        return "—"
    if isinstance(num, (int, float)):
        if abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        return f"{int(num):,}"
    return str(num)

def format_float(num, decimals=2):
    """Format float with specified decimals"""
    if pd.isna(num) or num is None:
        return "—"
    return f"{num:.{decimals}f}"

def create_download_link(content, filename="causal_report.md", link_text="Download Report"):
    """Create a download link"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/md;base64,{b64}" download="{filename}" style="text-decoration: none; color: #000000; font-weight: 500; border: 1px solid #e5e5e5; padding: 0.5rem 1rem; border-radius: 6px; display: inline-block; background: #ffffff;">{link_text}</a>'
    return href

def render_metric_card(label, value, unit=""):
    """Render a consistent metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """

def render_status_badge(text, type="info"):
    """Render a status badge"""
    badge_class = "status-badge"
    if type == "success":
        badge_class += " status-badge-success"
    elif type == "warning":
        badge_class += " status-badge-warning"
    elif type == "info":
        badge_class += " status-badge-info"
    
    return f'<span class="{badge_class}">{text}</span>'

if 'data' not in st.session_state:
    st.session_state.data = None
if 'true_params' not in st.session_state:
    st.session_state.true_params = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'config' not in st.session_state:
    st.session_state.config = {
        'data': DataConfig(),
        'model': ModelConfig(),
        'ui': UIConfig()
    }

def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Causal Effect Estimator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Estimate causal effects from observational data · MLE · IPTW · Causal Forest · Doubly Robust · Diagnostics</p>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        data_page = st.button("Data", use_container_width=True, 
                             type="primary" if st.session_state.get('current_page') == 'data' else "secondary")
        if data_page:
            st.session_state.current_page = 'data'
    
    with col2:
        model_page = st.button("Models", use_container_width=True,
                              type="primary" if st.session_state.get('current_page') == 'models' else "secondary")
        if model_page:
            st.session_state.current_page = 'models'
    
    with col3:
        eval_page = st.button("Evaluate", use_container_width=True,
                             type="primary" if st.session_state.get('current_page') == 'evaluate' else "secondary")
        if eval_page:
            st.session_state.current_page = 'evaluate'
    
    with col4:
        benchmark_page = st.button("Benchmark", use_container_width=True,
                                  type="primary" if st.session_state.get('current_page') == 'benchmark' else "secondary")
        if benchmark_page:
            st.session_state.current_page = 'benchmark'
    
    with col5:
        tutorial_page = st.button("Learn", use_container_width=True,
                                 type="primary" if st.session_state.get('current_page') == 'learn' else "secondary")
        if tutorial_page:
            st.session_state.current_page = 'learn'
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'data'
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    if st.session_state.current_page == 'data':
        show_data_generation()
    elif st.session_state.current_page == 'models':
        show_model_training()
    elif st.session_state.current_page == 'evaluate':
        show_evaluation()
    elif st.session_state.current_page == 'benchmark':
        show_benchmarking()
    elif st.session_state.current_page == 'learn':
        show_tutorial()
        
    st.markdown("""
    <div class="footer">
        <span style="font-weight: 500;">Causal Effect Estimator</span> · 
        MLE · IPTW · Causal Forest · Doubly Robust · 
        Built with Streamlit · Black & White Edition
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_generation():
    st.markdown('<div class="section-title">Data Generation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_source = "Synthetic"
        st.button("Synthetic", use_container_width=True,
                 type="primary" if st.session_state.get('data_source') == 'synthetic' else "secondary")
        if st.session_state.get('data_source') != 'synthetic':
            st.session_state.data_source = 'synthetic'
    
    with col2:
        st.button("Real Dataset", use_container_width=True,
                 type="primary" if st.session_state.get('data_source') == 'real' else "secondary")
    
    with col3:
        st.button("Upload CSV", use_container_width=True,
                 type="primary" if st.session_state.get('data_source') == 'upload' else "secondary")
    
    st.session_state.data_source = 'synthetic'
    
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-title">Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Sample size", 100, 10000, 2000, 100)
        treatment_effect = st.slider("True treatment effect", 0.0, 5.0, 2.0, 0.1)
        confounding_strength = st.slider("Confounding strength", 0.0, 3.0, 1.0, 0.1)
    
    with col2:
        n_features = st.slider("Number of features", 3, 20, 5)
        n_confounders = st.slider("Number of confounders", 1, n_features, 3)
        heterogeneity = st.checkbox("Heterogeneous effects", value=True)
        nonlinearity = st.selectbox("Nonlinearity", ["none", "moderate", "high"], index=0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("Generate Synthetic Data", use_container_width=True)
    
    if generate_button:
        with st.spinner("Generating data with known ground truth..."):
            config = DataConfig(
                n_samples=n_samples,
                n_features=n_features,
                n_confounders=n_confounders,
                treatment_effect=treatment_effect,
                effect_heterogeneity=heterogeneity,
                confounding_strength=confounding_strength,
                nonlinearity=nonlinearity
            )
            
            generator = CausalDataGenerator(config)
            df, true_params = generator.generate()
            
            st.session_state.data = df
            st.session_state.true_params = true_params
            st.session_state.trained_models = {}
            st.session_state.results = {}
            
            st.markdown(render_status_badge(f"✓ Generated {len(df):,} samples with ATE = {true_params['ate']:.3f}", "success"), 
                       unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        st.markdown('<div class="subsection-title">Data Preview</div>', unsafe_allow_html=True)
        
        df = st.session_state.data
        
        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_metric_card("Samples", format_number(len(df))), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card("Treatment Rate", f"{df['treatment'].mean():.1%}"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("Features", format_number(len(df.columns) - 2)), unsafe_allow_html=True)
        with col4:
            if st.session_state.true_params and 'ate' in st.session_state.true_params:
                st.markdown(render_metric_card("True ATE", format_float(st.session_state.true_params['ate'], 3)), 
                          unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("View raw data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

def show_model_training():
    st.markdown('<div class="section-title">Model Training</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.markdown("""
        <div class="info-message">
            <strong>No data available.</strong> Please generate or load data first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    X = df.drop(['treatment', 'outcome'], axis=1, errors='ignore')
    if 'propensity_score' in X.columns:
        X = X.drop('propensity_score', axis=1)
    
    T = df['treatment'].values
    Y = df['outcome'].values
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="subsection-title">Model Selection</div>', unsafe_allow_html=True)
        
        st.markdown("**Traditional Methods**")
        run_ols = st.checkbox("Linear Regression", value=True)
        run_iptw = st.checkbox("IPTW", value=True)
        run_strat = st.checkbox("Stratification", value=False)
        
        st.markdown("**Tree-Based Methods**")
        run_cf = st.checkbox("Causal Forest", value=True)
        run_slearner = st.checkbox("S-Learner", value=True)
        run_tlearner = st.checkbox("T-Learner", value=True)
        run_xlearner = st.checkbox("X-Learner", value=False)
        
        st.markdown("**Robust Methods**")
        run_dr = st.checkbox("Doubly Robust", value=True)
        run_aipw = st.checkbox("AIPW", value=False)
        
        st.markdown('<div class="subsection-title">Hyperparameters</div>', unsafe_allow_html=True)
        
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        n_estimators = st.slider("Number of trees", 50, 500, 100, 50)
        max_depth = st.slider("Max depth", 3, 15, 5)
        
        train_button = st.button("Train Selected Models", use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-title">Training Results</div>', unsafe_allow_html=True)
        
        if train_button:
            st.session_state.trained_models = {}
            st.session_state.results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_train = []
            
            if run_ols:
                models_to_train.append(("Linear Regression", LinearRegressionEstimator))
            if run_iptw:
                models_to_train.append(("IPTW", IPTWEstimator))
            if run_strat:
                models_to_train.append(("Stratification", StratificationEstimator))
            if run_cf:
                models_to_train.append(("Causal Forest", 
                    lambda: CausalForest(n_estimators=n_estimators, max_depth=max_depth)))
            if run_slearner:
                models_to_train.append(("S-Learner", SLearner))
            if run_tlearner:
                models_to_train.append(("T-Learner", TLearner))
            if run_xlearner:
                models_to_train.append(("X-Learner", XLearner))
            if run_dr:
                models_to_train.append(("Doubly Robust", 
                    lambda: DoublyRobustEstimator(n_folds=cv_folds)))
            if run_aipw:
                models_to_train.append(("AIPW", 
                    lambda: AIPWEstimator(n_folds=cv_folds)))
            
            for i, (name, estimator_class) in enumerate(models_to_train):
                status_text.text(f"Training {name}...")
                
                try:
                    estimator = estimator_class() if callable(estimator_class) else estimator_class()
                    estimator.fit(X, T, Y)
                    
                    st.session_state.trained_models[name] = estimator
                    
                    result = {
                        'ate': estimator.estimate_ate(),
                        'ites': estimator.get_ite() if hasattr(estimator, 'get_ite') else None
                    }
                    
                    if hasattr(estimator, 'estimate_ate_confidence_interval'):
                        try:
                            ci_lower, ci_upper = estimator.estimate_ate_confidence_interval()
                            result['ci_lower'] = ci_lower
                            result['ci_upper'] = ci_upper
                        except:
                            pass
                    
                    st.session_state.results[name] = result
                    
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)[:50]}...")
                
                progress_bar.progress((i + 1) / len(models_to_train))
            
            status_text.text("✓ Training complete!")
            progress_bar.empty()
        
        if st.session_state.trained_models:
            results_list = []
            for name, res in st.session_state.results.items():
                results_list.append({
                    'Estimator': name,
                    'ATE': f"{res['ate']:.3f}",
                    'CI': f"[{res.get('ci_lower', 0):.2f}, {res.get('ci_upper', 0):.2f}]" if 'ci_lower' in res else '—',
                })
            
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            if st.session_state.true_params and 'ate' in st.session_state.true_params:
                true_ate = st.session_state.true_params['ate']
                
                errors = {}
                for name, res in st.session_state.results.items():
                    errors[name] = abs(res['ate'] - true_ate)
                
                if errors:
                    best_model = min(errors, key=errors.get)
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">Best Estimate</div>
                        <strong>{best_model}</strong> · Error = {errors[best_model]:.3f} · True ATE = {true_ate:.3f}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Select models and click 'Train Selected Models'")

def show_evaluation():
    st.markdown('<div class="section-title">Evaluation & Interpretation</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.markdown("""
        <div class="info-message">
            <strong>No trained models.</strong> Please train models first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    true_params = st.session_state.true_params
    trained_models = st.session_state.trained_models
    results = st.session_state.results
    
    X = df.drop(['treatment', 'outcome'], axis=1, errors='ignore')
    if 'propensity_score' in X.columns:
        X = X.drop('propensity_score', axis=1)
    
    T = df['treatment'].values
    Y = df['outcome'].values
    
    tab1, tab2, tab3 = st.tabs(["Performance", "Diagnostics", "Individual Effects"])
    
    with tab1:
        if true_params and 'ate' in true_params:
            true_ate = true_params['ate']
            
            st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
            
            metrics_list = []
            for name, res in results.items():
                error = abs(res['ate'] - true_ate)
                bias = res['ate'] - true_ate
                metrics_list.append({
                    'name': name,
                    'error': error,
                    'bias': bias
                })
            
            metrics_df = pd.DataFrame(metrics_list)
            
            if not metrics_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    best_model = metrics_df.loc[metrics_df['error'].idxmin(), 'name']
                    st.markdown(render_metric_card("Best Model", best_model[:15]), unsafe_allow_html=True)
                
                with col2:
                    min_error = metrics_df['error'].min()
                    st.markdown(render_metric_card("Min Error", format_float(min_error, 3)), unsafe_allow_html=True)
                
                with col3:
                    avg_error = metrics_df['error'].mean()
                    st.markdown(render_metric_card("Avg Error", format_float(avg_error, 3)), unsafe_allow_html=True)
                
                with col4:
                    median_error = metrics_df['error'].median()
                    st.markdown(render_metric_card("Median Error", format_float(median_error, 3)), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="subsection-title">ATE Estimates vs Ground Truth</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            y_pos = list(range(len(results)))
            names = list(results.keys())
            ates = [res['ate'] for res in results.values()]
            
            fig.add_trace(go.Bar(
                y=names,
                x=ates,
                orientation='h',
                marker=dict(color=['#222222' for _ in names]),
                name='Estimated ATE',
                text=[f"{ate:.2f}" for ate in ates],
                textposition='outside'
            ))
            
            fig.add_vline(
                x=true_ate,
                line_dash="dash",
                line_color="#666666",
                annotation_text="True ATE",
                annotation_position="top"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter', size=11),
                xaxis=dict(
                    title="Average Treatment Effect",
                    gridcolor='#f0f0f0',
                    zeroline=True,
                    zerolinecolor='#e5e5e5'
                ),
                yaxis=dict(
                    gridcolor='#f0f0f0',
                    autorange="reversed"
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="subsection-title">Error Analysis</div>', unsafe_allow_html=True)
            
            error_df = metrics_df[['name', 'bias', 'error']].copy()
            error_df.columns = ['Estimator', 'Bias', 'Absolute Error']
            error_df = error_df.sort_values('Absolute Error').round(4)
            
            st.dataframe(error_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown('<div class="subsection-title">Causal Assumption Checks</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Overlap / Positivity**")
            
            ps = None
            if 'IPTW' in trained_models:
                ps = trained_models['IPTW'].propensity_scores
            elif 'Doubly Robust' in trained_models:
                ps = trained_models['Doubly Robust'].propensity_scores
            
            if ps is not None:
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=ps[T == 1],
                    name='Treated',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='#222222'
                ))
                
                fig.add_trace(go.Histogram(
                    x=ps[T == 0],
                    name='Control',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='#999999'
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    barmode='overlay',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Inter', size=11),
                    xaxis=dict(
                        title="Propensity Score",
                        gridcolor='#f0f0f0'
                    ),
                    yaxis=dict(
                        title="Count",
                        gridcolor='#f0f0f0'
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                overlap_stats = CausalDiagnostics.check_overlap(ps, T)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Min PS (Treated)", f"{overlap_stats['min_treated']:.3f}")
                with col_b:
                    st.metric("Max PS (Control)", f"{overlap_stats['max_control']:.3f}")
                with col_c:
                    st.metric("Violation Ratio", f"{overlap_stats['violation_ratio']:.1%}")
            else:
                st.info("Train IPTW or Doubly Robust to see overlap diagnostics")
        
        with col2:
            st.markdown("**Covariate Balance**")
            
            if ps is not None:
                weights = trained_models['IPTW'].weights if 'IPTW' in trained_models else None
                balance_df = CausalDiagnostics.check_covariate_balance(X, T, weights)
                
                if not balance_df.empty:
                    plot_df = balance_df.head(10).copy()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=plot_df['smd_before'],
                        y=plot_df['covariate'],
                        mode='markers',
                        name='Before',
                        marker=dict(color='#222222', size=8)
                    ))
                    
                    if weights is not None:
                        fig.add_trace(go.Scatter(
                            x=plot_df['smd_after'],
                            y=plot_df['covariate'],
                            mode='markers',
                            name='After',
                            marker=dict(color='#999999', size=8, symbol='diamond')
                        ))
                    
                    fig.add_vline(x=0.1, line_dash="dash", line_color="#666666", opacity=0.5)
                    fig.add_vline(x=-0.1, line_dash="dash", line_color="#666666", opacity=0.5)
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family='Inter', size=11),
                        xaxis=dict(
                            title="Standardized Mean Difference",
                            gridcolor='#f0f0f0'
                        ),
                        yaxis=dict(
                            title="",
                            gridcolor='#f0f0f0'
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    imbalanced_before = np.sum(np.abs(balance_df['smd_before']) > 0.1)
                    st.metric("Imbalanced Covariates", imbalanced_before)
            else:
                st.info("Train IPTW to see covariate balance")
    
    with tab3:
        st.markdown('<div class="subsection-title">Individual Treatment Effects (CATE)</div>', unsafe_allow_html=True)
        
        ite_dict = {}
        for name, model in trained_models.items():
            try:
                ites = model.get_ite() if hasattr(model, 'get_ite') else None
                if ites is not None:
                    ite_dict[name] = ites
            except:
                pass
        
        if ite_dict:
            fig = go.Figure()
            
            for name, ites in list(ite_dict.items())[:4]:
                fig.add_trace(go.Violin(
                    y=ites,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    line_color='#000000',
                    fillcolor='#f0f0f0',
                    opacity=0.8
                ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter', size=11),
                yaxis=dict(
                    title="Treatment Effect",
                    gridcolor='#f0f0f0'
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            summary_data = []
            for name, ites in ite_dict.items():
                summary_data.append({
                    'Model': name,
                    'Mean': f"{np.mean(ites):.2f}",
                    'Std': f"{np.std(ites):.2f}",
                    'Min': f"{np.min(ites):.2f}",
                    'Max': f"{np.max(ites):.2f}",
                    '% Positive': f"{np.mean(ites > 0) * 100:.0f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No models with CATE estimation available. Train Causal Forest or Metalearners.")

def show_benchmarking():
    st.markdown('<div class="section-title">Benchmarking</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-message">
        <strong>Compare estimators across different sample sizes.</strong>
        This will run multiple repetitions to assess performance stability.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subsection-title">Configuration</div>', unsafe_allow_html=True)
        
        n_repetitions = st.slider("Repetitions", 5, 50, 10, 5)
        
        sample_sizes = st.multiselect(
            "Sample sizes",
            [500, 1000, 2000, 5000, 10000],
            default=[500, 1000, 2000]
        )
        
        methods = st.multiselect(
            "Methods",
            ["Linear Regression", "IPTW", "Causal Forest", "S-Learner", "T-Learner", "Doubly Robust"],
            default=["Linear Regression", "Causal Forest", "Doubly Robust"]
        )
        
        run_benchmark = st.button("Run Benchmark", use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-title">Results</div>', unsafe_allow_html=True)
        
        if run_benchmark and sample_sizes and methods:
            st.info("Benchmark running... This may take a moment.")
            st.markdown("""
            <div style="background: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px; padding: 2rem; text-align: center;">
                <span style="color: #666666;">Benchmark results will appear here</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Configure and run benchmark")

def show_tutorial():
    st.markdown('<div class="section-title">Understanding Causal Inference</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">The Fundamental Problem</div>
        We can never observe both potential outcomes (Y₁ and Y₀) for the same unit.<br>
        <strong>Goal:</strong> Estimate E[Y₁ - Y₀] from observational data.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Assumptions**")
        st.markdown("""
        • **Unconfoundedness** — No unmeasured confounders  
        • **Positivity** — Everyone has chance of both treatments  
        • **Consistency** — Observed outcome = potential outcome  
        • **Non-interference** — No spillover effects
        """)
    
    with col2:
        st.markdown("**Estimation Methods**")
        st.markdown("""
        • **Linear Regression** — Control for confounders  
        • **IPTW** — Weight by inverse propensity  
        • **Causal Forest** — Non-linear CATE  
        • **Doubly Robust** — Two chances to be correct
        """)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px; padding: 1.5rem;">
        <strong>When to use each method:</strong><br><br>
        
        <strong>Linear Regression:</strong> Relationship is approximately linear, interpretability needed<br>
        <strong>IPTW:</strong> Trust propensity model, want to check covariate balance<br>
        <strong>Causal Forest:</strong> Need heterogeneous effects, non-linear relationships<br>
        <strong>Doubly Robust:</strong> Unsure about models, want protection against misspecification
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
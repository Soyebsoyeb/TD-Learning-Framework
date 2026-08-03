"""
TD Learning Framework - Professional Research Dashboard
Enhanced with better performance, more features, and cleaner architecture
Run with: streamlit run visualization/dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time
from datetime import datetime
import json
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.random_walk import RandomWalk
from environments.gridworld import GridWorld
from environments.cliff_walking import CliffWalking
from environments.windy_gridworld import WindyGridWorld

from algorithms.td0 import TD0
from algorithms.n_step_td import NStepTD
from algorithms.td_lambda import TDLambda
from algorithms.true_online_td_lambda import TrueOnlineTDLambda
from algorithms.sarsa import SARSA
from algorithms.expected_sarsa import ExpectedSARSA
from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning
from algorithms.watkins_q_lambda import WatkinsQLambda

st.set_page_config(
    page_title="TD Learning Research Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
ENV_MAP = {
    'Random Walk (5 states)': lambda: RandomWalk(5),
    'Random Walk (19 states)': lambda: RandomWalk(19),
    'GridWorld (5x5)': lambda: GridWorld(5, 5),
    'GridWorld (10x10)': lambda: GridWorld(10, 10),
    'Cliff Walking': CliffWalking,
    'Windy GridWorld': WindyGridWorld,
}

PREDICTION_ALGOS = {
    'TD(0)': lambda s, a, alpha, gamma, lam, eps: TD0(s, alpha=alpha, gamma=gamma),
    'n-Step TD (n=3)': lambda s, a, alpha, gamma, lam, eps: NStepTD(s, n=3, alpha=alpha, gamma=gamma),
    'n-Step TD (n=5)': lambda s, a, alpha, gamma, lam, eps: NStepTD(s, n=5, alpha=alpha, gamma=gamma),
    'TD(lambda)': lambda s, a, alpha, gamma, lam, eps: TDLambda(s, alpha=alpha, gamma=gamma, lambda_=lam),
    'True Online TD(lambda)': lambda s, a, alpha, gamma, lam, eps: TrueOnlineTDLambda(s, alpha=alpha, gamma=gamma, lambda_=lam),
}

CONTROL_ALGOS = {
    'SARSA': lambda s, a, alpha, gamma, lam, eps: SARSA(s, a, alpha=alpha, gamma=gamma, epsilon=eps),
    'Expected SARSA': lambda s, a, alpha, gamma, lam, eps: ExpectedSARSA(s, a, alpha=alpha, gamma=gamma, epsilon=eps),
    'Q-Learning': lambda s, a, alpha, gamma, lam, eps: QLearning(s, a, alpha=alpha, gamma=gamma, epsilon=eps),
    'Double Q-Learning': lambda s, a, alpha, gamma, lam, eps: DoubleQLearning(s, a, alpha=alpha, gamma=gamma, epsilon=eps),
    'Watkins Q(lambda)': lambda s, a, alpha, gamma, lam, eps: WatkinsQLambda(s, a, alpha=alpha, gamma=gamma, lambda_=lam, epsilon=eps),
}

# ============================================================================
# ENHANCED CSS
# ============================================================================
st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    .stApp { background: #0f0f1a; color: #e8e8e8; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== MAIN CONTAINER ===== */
    .main-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; margin: 0 auto; }
    
    /* ===== TOP BAR ===== */
    .top-bar {
        background: #1a1a2e;
        border-bottom: 1px solid #2a2a4a;
        padding: 0.75rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: -1rem -2rem 2rem -2rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.4);
    }
    .top-bar-brand { font-size: 1.25rem; font-weight: 600; color: #e8e8e8; letter-spacing: -0.5px; }
    .top-bar-brand span { color: #60a5fa; }
    .top-bar-status {
        font-size: 0.75rem;
        color: #94a3b8;
        background: #2a2a4a;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        border: 1px solid #3a3a5a;
    }
    .top-bar-status .dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #34d399;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg {
        background: #12121f !important;
        border-right: 1px solid #1f1f3a !important;
        padding: 0.5rem 0.5rem !important;
    }
    .sidebar-brand {
        padding: 0.5rem 0.75rem 1rem 0.75rem;
        border-bottom: 1px solid #1f1f3a;
        margin-bottom: 0.5rem;
    }
    .sidebar-brand h2 { color: #e8e8e8; font-size: 1.1rem; font-weight: 600; margin: 0; letter-spacing: -0.3px; }
    .sidebar-brand h2 span { color: #60a5fa; }
    .sidebar-brand p { color: #64748b; font-size: 0.65rem; margin: 0.15rem 0 0 0; letter-spacing: 0.3px; }
    
    .sidebar-section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem 0.25rem 0.75rem;
        margin-top: 0.25rem;
    }
    .sidebar-section-header .icon { font-size: 0.85rem; color: #60a5fa; }
    .sidebar-section-header .label {
        font-size: 0.6rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .sidebar-section-header .badge {
        font-size: 0.5rem;
        background: #2a2a4a;
        color: #94a3b8;
        padding: 0.1rem 0.4rem;
        border-radius: 10px;
        margin-left: auto;
    }
    .sidebar-section {
        background: #18182a;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.15rem 0.5rem 0.4rem 0.5rem;
        border: 1px solid #1f1f3a;
    }
    .sidebar-section .stSelectbox label,
    .sidebar-section .stSlider label,
    .sidebar-section .stNumberInput label,
    .sidebar-section .stRadio label {
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .sidebar-section .stSelectbox > div,
    .sidebar-section .stNumberInput > div {
        background: #0f0f1a !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 6px !important;
    }
    .sidebar-section .stSelectbox > div:hover,
    .sidebar-section .stNumberInput > div:hover {
        border-color: #3b82f6 !important;
    }
    .sidebar-section .stSlider > div > div { background: #1f1f3a !important; }
    .sidebar-section .stSlider > div > div > div { background: #3b82f6 !important; }
    .sidebar-section .stRadio > div { gap: 0.5rem !important; }
    .sidebar-section .stRadio > div > label {
        font-size: 0.75rem !important;
        color: #94a3b8 !important;
        padding: 0.15rem 0.5rem !important;
        border-radius: 4px !important;
        transition: all 0.2s !important;
    }
    .sidebar-section .stRadio > div > label:hover {
        background: #2a2a4a !important;
        color: #e8e8e8 !important;
    }
    .sidebar-section .stRadio > div > label > div { border-color: #3a3a5a !important; }
    .sidebar-section .stRadio > div > label > div[data-checked="true"] {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    .sidebar-section .stNumberInput > div > input {
        background: #0f0f1a !important;
        color: #e8e8e8 !important;
        border: none !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    
    /* ===== BUTTONS ===== */
    .run-button-container { padding: 0.25rem 0.5rem; margin-top: 0.25rem; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.3s !important;
        width: 100% !important;
        cursor: pointer !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }
    
    /* ===== STATUS CARD ===== */
    .status-card {
        background: #18182a;
        border: 1px solid #1f1f3a;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0.5rem;
    }
    .status-card .label { font-size: 0.55rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-card .value {
        font-size: 0.75rem;
        color: #e8e8e8;
        font-weight: 500;
        margin-top: 0.1rem;
    }
    .status-card .value .dot-online {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #34d399;
        border-radius: 50%;
        margin-right: 4px;
        animation: pulse-green 2s infinite;
    }
    
    /* ===== SIDEBAR FOOTER ===== */
    .sidebar-footer {
        padding: 0.5rem 0.75rem;
        margin-top: 0.25rem;
        border-top: 1px solid #1f1f3a;
    }
    .sidebar-footer .version { font-size: 0.55rem; color: #64748b; letter-spacing: 0.5px; }
    .sidebar-footer .key-shortcuts { font-size: 0.5rem; color: #4a4a6a; margin-top: 0.1rem; }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        transition: all 0.2s;
    }
    .metric-card:hover {
        border-color: #4a4a6a;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .metric-label { font-size: 0.7rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.3px; }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #e8e8e8;
        margin-top: 0.25rem;
        font-feature-settings: "tnum";
    }
    .metric-value.green { color: #34d399; }
    .metric-value.blue { color: #60a5fa; }
    .metric-value.orange { color: #fbbf24; }
    .metric-value.purple { color: #a78bfa; }
    .metric-value.red { color: #f87171; }
    .metric-change { font-size: 0.75rem; margin-top: 0.25rem; }
    .metric-change.up { color: #34d399; }
    .metric-change.down { color: #f87171; }
    .metric-change.neutral { color: #94a3b8; }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.25rem;
        margin-bottom: 1rem;
        border: 1px solid #2a2a4a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 0.8rem;
        color: #94a3b8;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.05);
        color: #e8e8e8;
    }
    .stTabs [aria-selected="true"] {
        background: #2a2a4a;
        color: #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* ===== WELCOME BOX ===== */
    .welcome-container { display: flex; justify-content: center; align-items: center; min-height: 60vh; }
    .welcome-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 3rem 4rem;
        max-width: 560px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .welcome-box h2 { color: #e8e8e8; font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0; letter-spacing: -0.5px; }
    .welcome-box p { color: #94a3b8; font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0 1.5rem 0; }
    .welcome-steps {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        text-align: left;
        margin: 1.5rem 0;
    }
    .welcome-step {
        background: #22223a;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border: 1px solid #2a2a4a;
    }
    .welcome-step .num { font-size: 0.6rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .welcome-step .text { font-size: 0.85rem; color: #e8e8e8; margin-top: 0.15rem; }
    .welcome-step .text.highlight { color: #60a5fa; font-weight: 500; }
    
    /* ===== STATUS BAR ===== */
    .status-bar {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin-top: 1.5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #64748b;
    }
    .status-bar span { color: #94a3b8; }
    
    /* ===== PLOTLY ===== */
    .js-plotly-plot {
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        background: #1a1a2e;
        padding: 0.5rem;
    }
    
    /* ===== DATAFRAME ===== */
    .dataframe {
        border: 1px solid #2a2a4a !important;
        border-radius: 6px !important;
        background: #1a1a2e !important;
    }
    .dataframe thead tr th {
        background: #22223a !important;
        color: #e8e8e8 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #2a2a4a !important;
    }
    .dataframe tbody tr td {
        border-bottom: 1px solid #2a2a4a !important;
        color: #94a3b8 !important;
    }
    .dataframe tbody tr:hover td { background: #22223a !important; }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a5a; }
    
    /* ===== PAGE TITLE ===== */
    .page-title { font-size: 1.5rem; font-weight: 600; color: #e8e8e8; margin-bottom: 0.25rem; letter-spacing: -0.5px; }
    .page-subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
class SessionState:
    """Manage session state with defaults."""
    
    @staticmethod
    def init():
        """Initialize session state variables."""
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'training_done' not in st.session_state:
            st.session_state.training_done = False
        if 'experiment_history' not in st.session_state:
            st.session_state.experiment_history = []
        if 'current_config' not in st.session_state:
            st.session_state.current_config = {}
    
    @staticmethod
    def save_experiment(config: Dict[str, Any], results: Dict[str, Any]):
        """Save experiment to history."""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': {
                'avg_reward': results.get('avg_reward', 0),
                'best_reward': results.get('best_reward', 0),
                'training_time': results.get('training_time', 0),
                'episodes': results.get('episodes', 0),
            }
        }
        st.session_state.experiment_history.append(experiment)
        # Keep last 20 experiments
        if len(st.session_state.experiment_history) > 20:
            st.session_state.experiment_history = st.session_state.experiment_history[-20:]

SessionState.init()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_environment(env_name: str):
    """Create environment from name."""
    if 'Random Walk' in env_name:
        env = ENV_MAP[env_name]()
        n_states = env.n_states + 2
        n_actions = getattr(env, 'n_actions', 2)
    else:
        env = ENV_MAP[env_name]()
        n_states = env.n_states if hasattr(env, 'n_states') else env.n_states
        n_actions = getattr(env, 'n_actions', 2)
    return env, n_states, n_actions

def get_algorithm(algo_type: str, algo_name: str, n_states: int, n_actions: int, 
                  alpha: float, gamma: float, lambda_: float, epsilon: float):
    """Get algorithm instance."""
    if algo_type == "Prediction":
        return PREDICTION_ALGOS[algo_name](n_states, n_actions, alpha, gamma, lambda_, epsilon)
    else:
        return CONTROL_ALGOS[algo_name](n_states, n_actions, alpha, gamma, lambda_, epsilon)

def compute_metrics(rewards: list) -> Dict[str, Any]:
    """Compute comprehensive metrics from rewards."""
    if len(rewards) == 0:
        return {}
    
    metrics = {
        'total_episodes': len(rewards),
        'total_reward': np.sum(rewards),
        'mean_reward': np.mean(rewards),
        'median_reward': np.median(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
    }
    
    if len(rewards) >= 50:
        metrics['mean_reward_last_50'] = np.mean(rewards[-50:])
    
    if len(rewards) >= 100:
        prev_50 = np.mean(rewards[-100:-50])
        curr_50 = np.mean(rewards[-50:])
        metrics['improvement'] = ((curr_50 - prev_50) / (abs(prev_50) + 1e-6)) * 100
        metrics['convergence'] = 100 * (1 - np.std(rewards[-100:]) / (abs(np.mean(rewards[-100:])) + 1e-6))
    
    return metrics

# ============================================================================
# TOP NAVIGATION
# ============================================================================
st.markdown("""
<div class="top-bar">
    <div class="top-bar-brand">TD <span>Learning</span> Platform</div>
    <div class="top-bar-status">
        <span class="dot"></span> System Ready
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>TD <span>Learning</span></h2>
        <p>Research Platform v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Environment
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="icon">🌍</span>
        <span class="label">Environment</span>
        <span class="badge">1/6</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    env_name = st.selectbox(
        "Select Environment",
        list(ENV_MAP.keys()),
        key="env_select"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Algorithm Type
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="icon">⚙️</span>
        <span class="label">Algorithm Type</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    algo_type = st.radio(
        "Select Algorithm Type",
        ["Prediction", "Control"],
        horizontal=True,
        key="algo_type_radio"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Algorithm
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="icon">🧠</span>
        <span class="label">Algorithm</span>
        <span class="badge">9 total</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if algo_type == "Prediction":
        algo_name = st.selectbox(
            "Select Algorithm",
            list(PREDICTION_ALGOS.keys()),
            key="algo_select_pred"
        )
    else:
        algo_name = st.selectbox(
            "Select Algorithm",
            list(CONTROL_ALGOS.keys()),
            key="algo_select_control"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hyperparameters
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="icon">📐</span>
        <span class="label">Hyperparameters</span>
        <span class="badge">4</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    alpha = st.slider(
        "Learning Rate (α)",
        0.001, 1.0, 0.1, 0.001,
        format="%.3f",
        key="alpha_slider"
    )
    gamma = st.slider(
        "Discount Factor (γ)",
        0.0, 1.0, 0.95, 0.01,
        format="%.2f",
        key="gamma_slider"
    )
    lambda_ = st.slider(
        "Eligibility Trace (λ)",
        0.0, 1.0, 0.8, 0.01,
        format="%.2f",
        key="lambda_slider"
    )
    epsilon = st.slider(
        "Exploration (ε)",
        0.0, 1.0, 0.1, 0.01,
        format="%.2f",
        key="epsilon_slider"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="icon">🎯</span>
        <span class="label">Training</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    n_episodes = st.number_input(
        "Number of Episodes",
        10, 10000, 500, 50,
        key="episodes_input"
    )
    seed = st.number_input(
        "Random Seed",
        0, 10000, 42, 1,
        key="seed_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Run button
    st.markdown('<div class="run-button-container">', unsafe_allow_html=True)
    run_button = st.button("▶ Run Experiment", use_container_width=True, key="run_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset button
    if st.session_state.training_done:
        st.markdown('<div style="padding: 0 0.5rem; margin-top: 0.25rem;">', unsafe_allow_html=True)
        if st.button("⟳ New Experiment", use_container_width=True, key="reset_btn"):
            st.session_state.results = None
            st.session_state.training_done = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Status card
    st.markdown("""
    <div class="status-card">
        <div class="label">System Status</div>
        <div class="value">
            <span class="dot-online"></span> Online · Ready
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="sidebar-footer">
        <div class="version">TD Learning Framework v2.0</div>
        <div class="key-shortcuts">⌘ + Enter to run</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
if run_button:
    np.random.seed(seed)
    
    # Create environment and algorithm
    env, n_states, n_actions = create_environment(env_name)
    algo = get_algorithm(algo_type, algo_name, n_states, n_actions, alpha, gamma, lambda_, epsilon)
    
    # Train with progress
    with st.spinner("Training in progress..."):
        progress = st.progress(0)
        start = time.time()
        
        if algo_type == "Prediction":
            V = algo.train(env, n_episodes=n_episodes)
        else:
            Q = algo.train(env, n_episodes=n_episodes)
        
        elapsed = time.time() - start
        progress.progress(100)
    
    # Save results
    rewards = algo.episode_rewards
    metrics = compute_metrics(rewards)
    
    st.session_state.results = {
        'algo': algo,
        'env': env,
        'env_name': env_name,
        'algo_name': algo_name,
        'algo_type': algo_type,
        'elapsed': elapsed,
        'n_episodes': n_episodes,
        'seed': seed,
        'alpha': alpha,
        'gamma': gamma,
        'lambda_': lambda_,
        'epsilon': epsilon,
        'n_states': n_states,
        'n_actions': n_actions,
        'rewards': rewards,
        'metrics': metrics
    }
    st.session_state.training_done = True
    
    # Save to history
    SessionState.save_experiment(
        config={'env': env_name, 'algo': algo_name, 'type': algo_type},
        results={'avg_reward': metrics.get('mean_reward', 0), 
                 'best_reward': metrics.get('max_reward', 0),
                 'training_time': elapsed,
                 'episodes': n_episodes}
    )

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
if st.session_state.training_done and st.session_state.results is not None:
    r = st.session_state.results
    algo = r['algo']
    env = r['env']
    rewards = r['rewards']
    metrics = r['metrics']
    
    # Page title
    st.markdown(f"""
    <div class="page-title">Experiment Results</div>
    <div class="page-subtitle">
        {r['algo_name']} · {r['env_name']} · {r['n_episodes']} episodes
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # METRIC CARDS
    # ========================================================================
    avg_reward = metrics.get('mean_reward_last_50', metrics.get('mean_reward', 0))
    best_reward = metrics.get('max_reward', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = metrics.get('improvement', 0)
        if 'improvement' in metrics:
            change_text = f"{'▲' if improvement > 0 else '▼'} {abs(improvement):.1f}% vs previous"
            change_class = "up" if improvement > 0 else "down"
        else:
            change_text = "─ Insufficient data"
            change_class = "neutral"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Reward (last 50)</div>
            <div class="metric-value green">{avg_reward:.2f}</div>
            <div class="metric-change {change_class}">{change_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Reward</div>
            <div class="metric-value blue">{best_reward:.2f}</div>
            <div class="metric-change neutral">Peak performance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Time</div>
            <div class="metric-value orange">{r['elapsed']:.2f}s</div>
            <div class="metric-change neutral">{r['n_episodes']} episodes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if hasattr(algo, 'episode_lengths') and len(algo.episode_lengths) > 0:
            avg_len = np.mean(algo.episode_lengths)
        else:
            avg_len = len(rewards)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Episode Length</div>
            <div class="metric-value purple">{avg_len:.1f}</div>
            <div class="metric-change neutral">Steps per episode</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Learning Curve", "📊 Value Function", "📋 Statistics", "📜 History"])
    
    # ------------------------------------------------------------------------
    # TAB 1: Learning Curve
    # ------------------------------------------------------------------------
    with tab1:
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.08,
                           row_heights=[0.65, 0.35])
        
        # Raw rewards
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards))),
                y=rewards,
                mode='lines',
                name='Reward',
                line=dict(color='#4a4a6a', width=1),
                opacity=0.5,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Smoothed
        fig.add_trace(
            go.Scatter(
                x=list(range(window-1, len(rewards))),
                y=smoothed,
                mode='lines',
                name='Smoothed',
                line=dict(color='#60a5fa', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(96,165,250,0.08)',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Cumulative average
        cum_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards))),
                y=cum_avg,
                mode='lines',
                name='Cumulative Avg',
                line=dict(color='#34d399', width=2, dash='dash'),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Final average
        final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        fig.add_hline(
            y=final_avg,
            line_dash="dot",
            line_color="#f87171",
            line_width=1.5,
            row=2, col=1,
            annotation_text=f"Final Avg: {final_avg:.2f}",
            annotation_font_color="#f87171",
            annotation_font_size=10
        )
        
        fig.update_layout(
            height=450,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='#94a3b8', size=11),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(26,26,46,0.9)',
                bordercolor='#2a2a4a',
                borderwidth=1
            ),
            margin=dict(l=50, r=30, t=30, b=40)
        )
        
        fig.update_xaxes(title_text="Episode", gridcolor='#2a2a4a', color='#64748b', row=1, col=1)
        fig.update_xaxes(title_text="Episode", gridcolor='#2a2a4a', color='#64748b', row=2, col=1)
        fig.update_yaxes(title_text="Reward", gridcolor='#2a2a4a', color='#64748b', row=1, col=1)
        fig.update_yaxes(title_text="Average Reward", gridcolor='#2a2a4a', color='#64748b', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Episodes", metrics.get('total_episodes', 0))
        c2.metric("Total Reward", f"{metrics.get('total_reward', 0):.2f}")
        c3.metric("Std Deviation", f"{metrics.get('std_reward', 0):.2f}")
        convergence = metrics.get('convergence', 0)
        c4.metric("Convergence", f"{convergence:.1f}%")
    
    # ------------------------------------------------------------------------
    # TAB 2: Value Function
    # ------------------------------------------------------------------------
    with tab2:
        if r['algo_type'] == "Prediction":
            V = algo.V
            
            if 'Random Walk' in r['env_name']:
                V_plot = V[1:-1]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V_plot))),
                    y=V_plot,
                    marker_color='#60a5fa',
                    marker_opacity=0.8,
                    text=[f'{v:.3f}' for v in V_plot],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=10)
                ))
                fig2.update_layout(
                    title="Value Function by State",
                    height=350,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#94a3b8', size=11),
                    margin=dict(l=40, r=40, t=50, b=40),
                    showlegend=False
                )
                fig2.update_xaxes(title_text="State", gridcolor='#2a2a4a', color='#64748b')
                fig2.update_yaxes(title_text="Value", gridcolor='#2a2a4a', color='#64748b')
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            elif hasattr(env, 'rows') and hasattr(env, 'cols'):
                grid = V.reshape(env.rows, env.cols)
                fig2 = go.Figure(data=go.Heatmap(
                    z=grid,
                    colorscale='Blues',
                    showscale=True,
                    text=[[f'{grid[i][j]:.2f}' for j in range(grid.shape[1])] for i in range(grid.shape[0])],
                    texttemplate='%{text}',
                    textfont={"size": 10, "color": "#e8e8e8"},
                    hoverongaps=False,
                    colorbar=dict(title="Value", tickfont=dict(color='#94a3b8'))
                ))
                fig2.update_layout(
                    title="Value Function Heatmap",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#94a3b8', size=11),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                fig2.update_xaxes(title_text="Column", gridcolor='#2a2a4a', color='#64748b')
                fig2.update_yaxes(title_text="Row", gridcolor='#2a2a4a', color='#64748b')
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V))),
                    y=V,
                    marker_color='#60a5fa',
                    marker_opacity=0.8
                ))
                fig2.update_layout(
                    title="Value Function",
                    height=350,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#94a3b8', size=11),
                    margin=dict(l=40, r=40, t=50, b=40),
                    showlegend=False
                )
                fig2.update_xaxes(title_text="State", gridcolor='#2a2a4a', color='#64748b')
                fig2.update_yaxes(title_text="Value", gridcolor='#2a2a4a', color='#64748b')
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        else:
            Q = algo.Q
            V = np.max(Q, axis=1)
            
            if 'Random Walk' in r['env_name']:
                V_plot = V[1:-1]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V_plot))),
                    y=V_plot,
                    marker_color='#34d399',
                    marker_opacity=0.8,
                    text=[f'{v:.3f}' for v in V_plot],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=10)
                ))
                fig2.update_layout(
                    title="Maximum Q-Values by State",
                    height=350,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#94a3b8', size=11),
                    margin=dict(l=40, r=40, t=50, b=40),
                    showlegend=False
                )
                fig2.update_xaxes(title_text="State", gridcolor='#2a2a4a', color='#64748b')
                fig2.update_yaxes(title_text="Max Q-Value", gridcolor='#2a2a4a', color='#64748b')
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V))),
                    y=V,
                    marker_color='#34d399',
                    marker_opacity=0.8
                ))
                fig2.update_layout(
                    title="Maximum Q-Values",
                    height=350,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#94a3b8', size=11),
                    margin=dict(l=40, r=40, t=50, b=40),
                    showlegend=False
                )
                fig2.update_xaxes(title_text="State", gridcolor='#2a2a4a', color='#64748b')
                fig2.update_yaxes(title_text="Max Q-Value", gridcolor='#2a2a4a', color='#64748b')
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    # ------------------------------------------------------------------------
    # TAB 3: Statistics
    # ------------------------------------------------------------------------
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Experiment Information")
            
            stats_metrics = [
                'Total Episodes',
                'Total Reward',
                'Average Reward',
                'Median Reward',
                'Std Deviation',
                'Min Reward',
                'Max Reward',
                'Training Time (s)',
                'Convergence Rate (%)'
            ]
            
            stats_values = [
                str(metrics.get('total_episodes', 0)),
                f"{metrics.get('total_reward', 0):.2f}",
                f"{metrics.get('mean_reward', 0):.2f}",
                f"{metrics.get('median_reward', 0):.2f}",
                f"{metrics.get('std_reward', 0):.2f}",
                f"{metrics.get('min_reward', 0):.2f}",
                f"{metrics.get('max_reward', 0):.2f}",
                f"{r['elapsed']:.2f}",
                f"{metrics.get('convergence', 0):.1f}"
            ]
            
            df_stats = pd.DataFrame({
                'Metric': stats_metrics,
                'Value': stats_values
            })
            
            st.dataframe(
                df_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small")
                }
            )
        
        with col2:
            st.markdown("### Configuration")
            
            config_metrics = [
                'Algorithm',
                'Type',
                'Environment',
                'Episodes',
                'Seed',
                'Learning Rate (α)',
                'Discount Factor (γ)',
                'Eligibility Trace (λ)',
                'Exploration (ε)'
            ]
            
            config_values = [
                r['algo_name'],
                r['algo_type'],
                r['env_name'],
                str(r['n_episodes']),
                str(r['seed']),
                f"{r['alpha']:.3f}",
                f"{r['gamma']:.2f}",
                f"{r['lambda_']:.2f}",
                f"{r['epsilon']:.2f}"
            ]
            
            df_config = pd.DataFrame({
                'Parameter': config_metrics,
                'Value': config_values
            })
            
            st.dataframe(
                df_config,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small")
                }
            )
        
        # Export
        st.markdown("### Export Results")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📥 Download CSV", use_container_width=True):
                export_data = pd.DataFrame({
                    'Episode': list(range(len(rewards))),
                    'Reward': rewards,
                    'Cumulative_Average': np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                })
                if hasattr(algo, 'episode_lengths'):
                    export_data['Episode_Length'] = algo.episode_lengths
                
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"td_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col2:
            if st.button("📊 View Raw Data", use_container_width=True):
                with st.expander("Raw Episode Data"):
                    st.dataframe(
                        pd.DataFrame({
                            'Episode': list(range(len(rewards))),
                            'Reward': rewards
                        }).head(20),
                        use_container_width=True
                    )
                    st.caption(f"Showing first 20 of {len(rewards)} episodes")
    
    # ------------------------------------------------------------------------
    # TAB 4: Experiment History
    # ------------------------------------------------------------------------
    with tab4:
        st.markdown("### Experiment History")
        
        if len(st.session_state.experiment_history) > 0:
            # Create history dataframe
            history_data = []
            for exp in st.session_state.experiment_history[-10:]:  # Show last 10
                history_data.append({
                    'Time': exp['timestamp'][:19],
                    'Algorithm': exp['config']['algo'],
                    'Environment': exp['config']['env'],
                    'Avg Reward': f"{exp['results']['avg_reward']:.2f}",
                    'Best Reward': f"{exp['results']['best_reward']:.2f}",
                    'Episodes': exp['results']['episodes'],
                    'Time (s)': f"{exp['results']['training_time']:.2f}"
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.experiment_history = []
                st.rerun()
        else:
            st.info("No experiments run yet. Run an experiment to see history here.")

# ============================================================================
# WELCOME SCREEN
# ============================================================================
else:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-box">
            <h2>🧪 TD Learning Research Platform</h2>
            <p>
                Configure your experiment using the sidebar controls, then click 
                <strong>"Run Experiment"</strong> to begin training.
            </p>
            <div class="welcome-steps">
                <div class="welcome-step">
                    <div class="num">Step 1</div>
                    <div class="text">Select environment</div>
                </div>
                <div class="welcome-step">
                    <div class="num">Step 2</div>
                    <div class="text">Choose algorithm</div>
                </div>
                <div class="welcome-step">
                    <div class="num">Step 3</div>
                    <div class="text">Set hyperparameters</div>
                </div>
                <div class="welcome-step">
                    <div class="num">Step 4</div>
                    <div class="text highlight">▶ Run Experiment</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                Supported: 6 environments · 9 algorithms
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# STATUS BAR
# ============================================================================
st.markdown("""
<div class="status-bar">
    <span>● System: Online</span>
    <span>Framework: TD Learning v2.0</span>
    <span>Status: Ready</span>
    <span>{}</span>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

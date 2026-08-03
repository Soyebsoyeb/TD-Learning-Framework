"""
Advanced HFT-Style Dashboard for TD Learning Framework.
Professional dark theme with real-time analytics and financial-style visualizations.
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

# Page configuration
st.set_page_config(
    page_title="TD Learning Framework",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HFT Style CSS
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: #0a0a0f;
        color: #e0e0e0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main-container {
        padding: 20px 30px;
        background: #0a0a0f;
        min-height: 100vh;
    }

    /* Header - HFT Style */
    .hft-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 20px 30px;
        border-radius: 8px;
        border: 1px solid #2a2a4a;
        margin-bottom: 25px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hft-header h1 {
        font-size: 2rem;
        font-weight: 300;
        color: #00d4ff;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0,212,255,0.3);
    }
    .hft-header .status {
        color: #00ff88;
        font-size: 0.85rem;
        padding: 5px 15px;
        border: 1px solid #00ff88;
        border-radius: 20px;
        background: rgba(0,255,136,0.05);
    }
    .hft-header .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    /* HFT Metric Cards */
    .hft-metric {
        background: #0d0d1a;
        border: 1px solid #1a1a3a;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 5px 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .hft-metric:hover {
        border-color: #00d4ff;
        box-shadow: 0 4px 25px rgba(0,212,255,0.1);
        transform: translateY(-2px);
    }
    .hft-metric-label {
        color: #8899aa;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .hft-metric-value {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 300;
        margin: 5px 0 0 0;
        font-family: 'Courier New', monospace;
    }
    .hft-metric-value.green {
        color: #00ff88;
    }
    .hft-metric-value.red {
        color: #ff4466;
    }
    .hft-metric-change {
        font-size: 0.75rem;
        margin-top: 3px;
    }
    .hft-metric-change.up {
        color: #00ff88;
    }
    .hft-metric-change.down {
        color: #ff4466;
    }

    /* Sidebar - HFT Style */
    .css-1d391kg {
        background: #0a0a12;
        border-right: 1px solid #1a1a3a;
    }
    .sidebar-section {
        background: #0d0d1a;
        border: 1px solid #1a1a3a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .sidebar-section-title {
        color: #8899aa;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        border-bottom: 1px solid #1a1a3a;
        padding-bottom: 8px;
    }

    /* Controls */
    .stSelectbox, .stSlider, .stNumberInput {
        background: transparent;
    }
    .stSelectbox > div, .stSlider > div, .stNumberInput > div {
        background: #0a0a12;
        border: 1px solid #1a1a3a;
        border-radius: 6px;
        color: #e0e0e0;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #8899aa !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Button - HFT Style */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0066ff 100%);
        color: #0a0a0f;
        border: none;
        padding: 12px 30px;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 4px 20px rgba(0,212,255,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,212,255,0.4);
        background: linear-gradient(135deg, #00e4ff 0%, #0077ff 100%);
    }

    /* Tabs - HFT Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #0a0a12;
        border-radius: 6px;
        padding: 4px;
        border: 1px solid #1a1a3a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px;
        padding: 8px 20px;
        font-weight: 500;
        color: #8899aa;
        transition: all 0.2s ease;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0,212,255,0.05);
        color: #00d4ff;
    }
    .stTabs [aria-selected="true"] {
        background: #00d4ff;
        color: #0a0a0f;
        box-shadow: 0 4px 15px rgba(0,212,255,0.3);
    }

    /* Plotly container */
    .js-plotly-plot {
        border: 1px solid #1a1a3a;
        border-radius: 8px;
        background: #0d0d1a;
        padding: 10px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a1a3a;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2a2a5a;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #0d0d1a;
        border: 1px solid #1a1a3a;
        border-radius: 6px;
        color: #8899aa;
        font-weight: 500;
    }
    .streamlit-expanderContent {
        background: #0a0a12;
        border: 1px solid #1a1a3a;
        border-radius: 0 0 6px 6px;
    }

    /* Info box */
    .hft-info {
        background: #0d0d1a;
        border: 1px solid #1a1a3a;
        border-left: 3px solid #00d4ff;
        padding: 12px 15px;
        border-radius: 4px;
        color: #8899aa;
        font-size: 0.8rem;
        margin: 10px 0;
    }

    /* Status bar */
    .status-bar {
        background: #0a0a12;
        border: 1px solid #1a1a3a;
        border-radius: 6px;
        padding: 8px 15px;
        margin-top: 20px;
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #556677;
    }
    .status-bar span {
        font-family: 'Courier New', monospace;
    }

    /* Welcome screen */
    .welcome-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
        flex-direction: column;
        text-align: center;
    }
    .welcome-box {
        background: #0d0d1a;
        border: 1px solid #1a1a3a;
        border-radius: 12px;
        padding: 50px;
        max-width: 700px;
    }
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        color: #00d4ff;
    }
    .welcome-title {
        color: #00d4ff;
        margin: 0;
        font-weight: 300;
        letter-spacing: 2px;
    }
    .welcome-text {
        color: #8899aa;
        font-size: 1.1rem;
        margin: 20px 0;
        line-height: 1.6;
    }
    .step-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 30px 0;
        text-align: left;
    }
    .step-item {
        background: #0a0a12;
        border: 1px solid #1a1a3a;
        border-radius: 8px;
        padding: 15px;
    }
    .step-number {
        color: #556677;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .step-desc {
        color: #e0e0e0;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .step-highlight {
        color: #00d4ff;
        font-size: 0.9rem;
        margin-top: 5px;
        font-weight: 600;
    }
    .welcome-footer {
        color: #556677;
        font-size: 0.85rem;
        margin-top: 20px;
        border-top: 1px solid #1a1a3a;
        padding-top: 20px;
    }
    .welcome-dot-green {
        color: #00ff88;
    }
    .welcome-dot-blue {
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Environment and algorithm mappings
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

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'algo' not in st.session_state:
    st.session_state.algo = None

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HFT Header
st.markdown("""
<div class="hft-header">
    <div>
        <h1>TD LEARNING FRAMEWORK</h1>
        <div style="font-size: 0.8rem; color: #556677; margin-top: 5px; letter-spacing: 1px;">
            TEMPORAL DIFFERENCE LEARNING · REAL-TIME ANALYTICS
        </div>
    </div>
    <div class="status">
        <span class="status-dot"></span>
        SYSTEM ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - HFT Style
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0;">
        <div style="color: #00d4ff; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 15px;">
            CONFIGURATION
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Environment
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">ENVIRONMENT</div>', unsafe_allow_html=True)
    env_name = st.selectbox(
        "Select Environment",
        list(ENV_MAP.keys()),
        key="env_select",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Algorithm Type
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">ALGORITHM TYPE</div>', unsafe_allow_html=True)
    algo_type = st.radio(
        "Select Algorithm Type",
        ["Prediction", "Control"],
        key="algo_type_radio",
        label_visibility="collapsed",
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Algorithm
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">ALGORITHM</div>', unsafe_allow_html=True)
    if algo_type == "Prediction":
        algo_name = st.selectbox(
            "Select Algorithm",
            list(PREDICTION_ALGOS.keys()),
            key="algo_select_pred",
            label_visibility="collapsed"
        )
    else:
        algo_name = st.selectbox(
            "Select Algorithm",
            list(CONTROL_ALGOS.keys()),
            key="algo_select_control",
            label_visibility="collapsed"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Hyperparameters
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">HYPERPARAMETERS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider(
            "Alpha",
            0.001, 1.0, 0.1, 0.001,
            format="%.3f",
            key="alpha_slider",
            label_visibility="collapsed"
        )
        st.caption("Learning Rate")
    with col2:
        gamma = st.slider(
            "Gamma",
            0.0, 1.0, 0.95, 0.01,
            format="%.2f",
            key="gamma_slider",
            label_visibility="collapsed"
        )
        st.caption("Discount Factor")

    lambda_ = st.slider(
        "Lambda",
        0.0, 1.0, 0.8, 0.01,
        format="%.2f",
        key="lambda_slider",
        label_visibility="collapsed"
    )
    st.caption("Eligibility Trace")

    epsilon = st.slider(
        "Epsilon",
        0.0, 1.0, 0.1, 0.01,
        format="%.2f",
        key="epsilon_slider",
        label_visibility="collapsed"
    )
    st.caption("Exploration Rate")
    st.markdown('</div>', unsafe_allow_html=True)

    # Training
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">TRAINING</div>', unsafe_allow_html=True)
    n_episodes = st.number_input(
        "Number of Episodes",
        10, 10000, 500, 10,
        key="episodes_input",
        label_visibility="collapsed"
    )
    seed = st.number_input(
        "Random Seed",
        0, 10000, 42, 1,
        key="seed_input",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Run button
    st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
    run_button = st.button("EXECUTE TRAINING", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # System info
    st.markdown("""
    <div class="hft-info" style="margin-top: 15px;">
        <div style="color: #556677; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;">
            System Status
        </div>
        <div style="color: #8899aa; font-size: 0.75rem; font-family: 'Courier New', monospace; margin-top: 5px;">
            ● Ready
        </div>
        <div style="color: #556677; font-size: 0.65rem; margin-top: 5px;">
            Framework v2.0 · Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content area - Training execution
if run_button:
    np.random.seed(seed)

    # Get environment info
    if 'Random Walk' in env_name:
        env = ENV_MAP[env_name]()
        n_states = env.n_states + 2
        n_actions = getattr(env, 'n_actions', 2)
    else:
        env = ENV_MAP[env_name]()
        n_states = env.n_states if hasattr(env, 'n_states') else env.n_states
        n_actions = getattr(env, 'n_actions', 2)

    # Initialize algorithm
    if algo_type == "Prediction":
        algo = PREDICTION_ALGOS[algo_name](n_states, n_actions, alpha, gamma, lambda_, epsilon)
    else:
        algo = CONTROL_ALGOS[algo_name](n_states, n_actions, alpha, gamma, lambda_, epsilon)

    # Training with progress
    with st.spinner("Training in progress..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        # Custom training with progress
        if algo_type == "Prediction":
            V = algo.train(env, n_episodes=n_episodes)
        else:
            Q = algo.train(env, n_episodes=n_episodes)

        elapsed_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.text("Training complete!")

    st.session_state.training_complete = True
    st.session_state.results = {
        'algo': algo,
        'env_name': env_name,
        'algo_name': algo_name,
        'algo_type': algo_type,
        'n_episodes': n_episodes,
        'elapsed_time': elapsed_time,
        'n_states': n_states,
        'env': env
    }
    st.session_state.algo = algo

# Display results or welcome screen
if st.session_state.training_complete and st.session_state.results is not None:
    results = st.session_state.results
    algo = results['algo']
    env = results['env']
    n_actions = getattr(env, 'n_actions', 2)

    # HFT Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_reward = np.mean(algo.episode_rewards[-50:])
        change = ((avg_reward - np.mean(algo.episode_rewards[-100:-50])) / (abs(np.mean(algo.episode_rewards[-100:-50])) + 1e-6) * 100) if len(algo.episode_rewards) >= 100 else 0
        st.markdown(f"""
        <div class="hft-metric">
            <div class="hft-metric-label">Average Reward (Last 50)</div>
            <div class="hft-metric-value green">{avg_reward:.2f}</div>
            <div class="hft-metric-change {'up' if change > 0 else 'down'}">{'+' if change > 0 else ''}{change:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        best_reward = np.max(algo.episode_rewards)
        st.markdown(f"""
        <div class="hft-metric">
            <div class="hft-metric-label">Best Reward</div>
            <div class="hft-metric-value">{best_reward:.2f}</div>
            <div class="hft-metric-change" style="color: #8899aa;">Peak Performance</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="hft-metric">
            <div class="hft-metric-label">Training Time</div>
            <div class="hft-metric-value">{results['elapsed_time']:.2f}s</div>
            <div class="hft-metric-change" style="color: #8899aa;">{algo_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if hasattr(algo, 'episode_lengths') and len(algo.episode_lengths) > 0:
            avg_length = np.mean(algo.episode_lengths)
        else:
            avg_length = len(algo.episode_rewards)
        st.markdown(f"""
        <div class="hft-metric">
            <div class="hft-metric-label">Avg Episode Length</div>
            <div class="hft-metric-value">{avg_length:.1f}</div>
            <div class="hft-metric-change" style="color: #8899aa;">Steps per episode</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Value Function", "Policy", "Statistics"])

    with tab1:
        st.markdown('<div style="margin-bottom: 15px; color: #8899aa; font-size: 0.8rem; letter-spacing: 1px;">REWARD PROGRESSION</div>', unsafe_allow_html=True)

        rewards = algo.episode_rewards
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("", ""),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )

        # Main chart - Dark theme
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards))),
                y=rewards,
                mode='lines',
                name='Raw',
                line=dict(color='#1a2a4a', width=1),
                opacity=0.5
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(window-1, len(rewards))),
                y=smoothed,
                mode='lines',
                name='Smoothed',
                line=dict(color='#00d4ff', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0,212,255,0.1)'
            ),
            row=1, col=1
        )

        # Cumulative average
        cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards))),
                y=cumulative_avg,
                mode='lines',
                name='Cumulative Avg',
                line=dict(color='#00ff88', width=2, dash='dash'),
            ),
            row=2, col=1
        )

        # Add horizontal line for final average
        final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        fig.add_hline(
            y=final_avg,
            line_dash="dot",
            line_color="#ff4466",
            line_width=1,
            row=2, col=1,
            annotation_text=f"Final Avg: {final_avg:.2f}",
            annotation_font_color="#ff4466"
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#8899aa', size=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=40, r=40, t=30, b=40)
        )

        fig.update_xaxes(
            title_text="Episode",
            gridcolor='#1a1a3a',
            color='#556677',
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Episode",
            gridcolor='#1a1a3a',
            color='#556677',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Reward",
            gridcolor='#1a1a3a',
            color='#556677',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Average Reward",
            gridcolor='#1a1a3a',
            color='#556677',
            row=2, col=1
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quick stats below chart
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Episodes", len(rewards))
        with col2:
            st.metric("Total Reward", f"{np.sum(rewards):.2f}")
        with col3:
            st.metric("Std Deviation", f"{np.std(rewards):.2f}")
        with col4:
            convergence = 100 * (1 - np.std(rewards[-100:]) / (abs(np.mean(rewards[-100:])) + 1e-6))
            st.metric("Convergence", f"{convergence:.1f}%")

    with tab2:
        st.markdown('<div style="margin-bottom: 15px; color: #8899aa; font-size: 0.8rem; letter-spacing: 1px;">VALUE FUNCTION ANALYSIS</div>', unsafe_allow_html=True)

        if algo_type == "Prediction":
            V = algo.V

            if 'Random Walk' in results['env_name']:
                V_plot = V[1:-1]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V_plot))),
                    y=V_plot,
                    marker_color='#00d4ff',
                    marker_opacity=0.8,
                    text=[f'{v:.3f}' for v in V_plot],
                    textposition='outside',
                    textfont=dict(color='#8899aa', size=9)
                ))
                fig2.update_layout(
                    title="Value Function by State",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig2.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                fig2.update_yaxes(title_text="Value", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig2, use_container_width=True)

            elif hasattr(env, 'rows') and hasattr(env, 'cols'):
                grid = V.reshape(env.rows, env.cols)
                fig2 = go.Figure(data=go.Heatmap(
                    z=grid,
                    colorscale='Viridis',
                    showscale=True,
                    text=[[f'{grid[i][j]:.2f}' for j in range(grid.shape[1])] for i in range(grid.shape[0])],
                    texttemplate='%{text}',
                    textfont={"size": 10, "color": "white"},
                    hoverongaps=False,
                    colorbar=dict(title="Value", tickfont=dict(color='#8899aa'))
                ))
                fig2.update_layout(
                    title="Value Function Heatmap",
                    height=450,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig2.update_xaxes(title_text="Column", gridcolor='#1a1a3a', color='#556677')
                fig2.update_yaxes(title_text="Row", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V))),
                    y=V,
                    marker_color='#00d4ff',
                    marker_opacity=0.8
                ))
                fig2.update_layout(
                    title="Value Function",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig2.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                fig2.update_yaxes(title_text="Value", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig2, use_container_width=True)

        else:
            Q = algo.Q
            V = np.max(Q, axis=1)

            if 'Random Walk' in results['env_name']:
                V_plot = V[1:-1]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V_plot))),
                    y=V_plot,
                    marker_color='#00ff88',
                    marker_opacity=0.8,
                    text=[f'{v:.3f}' for v in V_plot],
                    textposition='outside',
                    textfont=dict(color='#8899aa', size=9)
                ))
                fig2.update_layout(
                    title="Maximum Q-Values by State",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig2.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                fig2.update_yaxes(title_text="Max Q-Value", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig2, use_container_width=True)

            elif hasattr(env, 'rows') and hasattr(env, 'cols'):
                try:
                    grid = V.reshape(env.rows, env.cols)
                    fig2 = go.Figure(data=go.Heatmap(
                        z=grid,
                        colorscale='Viridis',
                        showscale=True,
                        text=[[f'{grid[i][j]:.2f}' for j in range(grid.shape[1])] for i in range(grid.shape[0])],
                        texttemplate='%{text}',
                        textfont={"size": 10, "color": "white"},
                        hoverongaps=False,
                        colorbar=dict(title="Q-Value", tickfont=dict(color='#8899aa'))
                    ))
                    fig2.update_layout(
                        title="Q-Value Heatmap",
                        height=450,
                        template='plotly_dark',
                        paper_bgcolor='#0a0a0f',
                        plot_bgcolor='#0a0a0f',
                        font=dict(color='#8899aa'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    fig2.update_xaxes(title_text="Column", gridcolor='#1a1a3a', color='#556677')
                    fig2.update_yaxes(title_text="Row", gridcolor='#1a1a3a', color='#556677')
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=list(range(len(V))),
                        y=V,
                        mode='lines+markers',
                        line=dict(color='#00ff88', width=2),
                        marker=dict(color='#00d4ff', size=6)
                    ))
                    fig2.update_layout(
                        title="Maximum Q-Values",
                        height=400,
                        template='plotly_dark',
                        paper_bgcolor='#0a0a0f',
                        plot_bgcolor='#0a0a0f',
                        font=dict(color='#8899aa'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    fig2.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                    fig2.update_yaxes(title_text="Max Q-Value", gridcolor='#1a1a3a', color='#556677')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(len(V))),
                    y=V,
                    marker_color='#00ff88',
                    marker_opacity=0.8
                ))
                fig2.update_layout(
                    title="Maximum Q-Values",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig2.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                fig2.update_yaxes(title_text="Max Q-Value", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown('<div style="margin-bottom: 15px; color: #8899aa; font-size: 0.8rem; letter-spacing: 1px;">POLICY ANALYSIS</div>', unsafe_allow_html=True)

        if algo_type == "Control" and hasattr(algo, 'get_policy'):
            policy = algo.get_policy()

            if 'Random Walk' in results['env_name']:
                policy_plot = policy[1:-1]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=list(range(len(policy_plot))),
                    y=policy_plot,
                    marker_color='#ff6b6b',
                    marker_opacity=0.8,
                    text=[f'Action {a}' for a in policy_plot],
                    textposition='outside',
                    textfont=dict(color='#8899aa', size=9)
                ))
                fig3.update_layout(
                    title="Policy by State",
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0a0f',
                    plot_bgcolor='#0a0a0f',
                    font=dict(color='#8899aa'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig3.update_xaxes(title_text="State", gridcolor='#1a1a3a', color='#556677')
                fig3.update_yaxes(title_text="Action", gridcolor='#1a1a3a', color='#556677')
                st.plotly_chart(fig3, use_container_width=True)

            elif hasattr(env, 'rows') and hasattr(env, 'cols'):
                try:
                    policy_grid = policy.reshape(env.rows, env.cols)

                    # Create policy heatmap
                    fig3 = go.Figure(data=go.Heatmap(
                        z=policy_grid,
                        colorscale='RdYlBu',
                        showscale=True,
                        text=[[f'{policy_grid[i][j]}' for j in range(policy_grid.shape[1])] for i in range(policy_grid.shape[0])],
                        texttemplate='%{text}',
                        textfont={"size": 12, "color": "white"},
                        hoverongaps=False,
                        colorbar=dict(title="Action", tickfont=dict(color='#8899aa'))
                    ))
                    fig3.update_layout(
                        title="Policy Heatmap",
                        height=450,
                        template='plotly_dark',
                        paper_bgcolor='#0a0a0f',
                        plot_bgcolor='#0a0a0f',
                        font=dict(color='#8899aa'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    fig3.update_xaxes(title_text="Column", gridcolor='#1a1a3a', color='#556677')
                    fig3.update_yaxes(title_text="Row", gridcolor='#1a1a3a', color='#556677')
                    st.plotly_chart(fig3, use_container_width=True)

                    # Action distribution
                    action_counts = np.bincount(policy.flatten(), minlength=n_actions)
                    fig3b = go.Figure(data=[go.Pie(
                        labels=[f'Action {i}' for i in range(n_actions)],
                        values=action_counts,
                        hole=0.4,
                        marker=dict(colors=['#00d4ff', '#00ff88', '#ff6b6b', '#ffd93d'][:n_actions]),
                        textfont=dict(color='#8899aa')
                    )])
                    fig3b.update_layout(
                        title="Action Distribution",
                        height=350,
                        template='plotly_dark',
                        paper_bgcolor='#0a0a0f',
                        plot_bgcolor='#0a0a0f',
                        font=dict(color='#8899aa'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig3b, use_container_width=True)
                except Exception as e:
                    st.warning(f"Policy visualization not available: {str(e)}")
            else:
                st.info("Policy visualization available for grid environments")
        else:
            st.info("Policy analysis available for Control algorithms")

    with tab4:
        st.markdown('<div style="margin-bottom: 15px; color: #8899aa; font-size: 0.8rem; letter-spacing: 1px;">DETAILED STATISTICS</div>', unsafe_allow_html=True)

        # Statistics table
        stats_data = {
            'Metric': [
                'Total Episodes',
                'Total Reward',
                'Average Reward',
                'Median Reward',
                'Std Deviation',
                'Min Reward',
                'Max Reward',
                'Training Time (s)',
                'Convergence Rate (%)'
            ],
            'Value': [
                len(algo.episode_rewards),
                np.sum(algo.episode_rewards),
                np.mean(algo.episode_rewards),
                np.median(algo.episode_rewards),
                np.std(algo.episode_rewards),
                np.min(algo.episode_rewards),
                np.max(algo.episode_rewards),
                results["elapsed_time"],
                100 * (1 - np.std(algo.episode_rewards[-100:]) / (abs(np.mean(algo.episode_rewards[-100:])) + 1e-6))
            ]
        }

        df_stats = pd.DataFrame(stats_data)
        df_stats_display = df_stats.copy()
        df_stats_display['Value'] = df_stats_display['Value'].apply(
            lambda x: f'{x:.2f}' if isinstance(x, float) else str(x)
        )

        st.dataframe(
            df_stats_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small")
            }
        )

        # Algorithm info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: #0d0d1a; border: 1px solid #1a1a3a; border-radius: 8px; padding: 15px; margin-top: 15px;">
                <div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">Algorithm Information</div>
                <div style="color: #e0e0e0; font-size: 0.85rem; margin-top: 8px; font-family: 'Courier New', monospace;">
                    <div>Name: {}</div>
                    <div>Type: {}</div>
                    <div>Environment: {}</div>
                    <div>Seed: {}</div>
                </div>
            </div>
            """.format(
                results['algo_name'],
                results['algo_type'],
                results['env_name'],
                seed
            ), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: #0d0d1a; border: 1px solid #1a1a3a; border-radius: 8px; padding: 15px; margin-top: 15px;">
                <div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">Hyperparameters</div>
                <div style="color: #e0e0e0; font-size: 0.85rem; margin-top: 8px; font-family: 'Courier New', monospace;">
                    <div>Alpha: {:.3f}</div>
                    <div>Gamma: {:.2f}</div>
                    <div>Lambda: {:.2f}</div>
                    <div>Epsilon: {:.2f}</div>
                </div>
            </div>
            """.format(alpha, gamma, lambda_, epsilon), unsafe_allow_html=True)

        # Export
        st.markdown("""
        <div style="margin-top: 20px; border-top: 1px solid #1a1a3a; padding-top: 20px;">
            <div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Export Results</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Download CSV", use_container_width=True):
            export_data = pd.DataFrame({
                'Episode': list(range(len(algo.episode_rewards))),
                'Reward': algo.episode_rewards,
                'Cumulative_Average': np.cumsum(algo.episode_rewards) / (np.arange(len(algo.episode_rewards)) + 1)
            })
            if hasattr(algo, 'episode_lengths'):
                export_data['Episode_Length'] = algo.episode_lengths

            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download",
                data=csv,
                file_name=f"td_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    # Welcome / Initial State
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-box">
            <div class="welcome-icon">🧠</div>
            <h2 class="welcome-title">TD LEARNING FRAMEWORK</h2>
            <p class="welcome-text">
                Ready to run experiments with Temporal Difference learning algorithms.
            </p>
            <div class="step-grid">
                <div class="step-item">
                    <div class="step-number">Step 1</div>
                    <div class="step-desc">Select environment and algorithm</div>
                </div>
                <div class="step-item">
                    <div class="step-number">Step 2</div>
                    <div class="step-desc">Configure hyperparameters</div>
                </div>
                <div class="step-item">
                    <div class="step-number">Step 3</div>
                    <div class="step-desc">Set training parameters</div>
                </div>
                <div class="step-item">
                    <div class="step-number">Step 4</div>
                    <div class="step-highlight">Click "EXECUTE TRAINING"</div>
                </div>
            </div>
            <div class="welcome-footer">
                <span class="welcome-dot-green">●</span> Configure parameters in the sidebar
                <span style="color: #8899aa; margin: 0 10px;">|</span>
                <span class="welcome-dot-blue">●</span> Results will appear here
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Status Bar
st.markdown("""
<div class="status-bar">
    <span>SYSTEM: ONLINE</span>
    <span>FRAMEWORK: TD LEARNING v2.0</span>
    <span>STATUS: READY</span>
    <span>TIMESTAMP: {}</span>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

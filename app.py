"""
Stock Price Scenario Simulator
==============================
Interactive Streamlit app that generates thousands of hypothetical
future stock price paths using Geometric Brownian Motion.

Optimized for Streamlit Community Cloud to prevent CPU throttling.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.gbm import simulate_gbm, analytical_mean, analytical_std
from src.risk import compute_risk_metrics
from config.settings import (
    APP_TITLE, APP_ICON, APP_CAPTION,
    DEFAULT_S0, DEFAULT_MU, DEFAULT_SIGMA, DEFAULT_T,
    DEFAULT_STEPS, DEFAULT_N_PATHS, DEFAULT_SEED, DEFAULT_RF,
    MAX_PATHS, MAX_DISPLAYED_PATHS,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(APP_CAPTION)

# ----------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------
st.sidebar.header("Simulation Parameters")

S0 = st.sidebar.number_input(
    "Initial Price (S₀)", min_value=0.01,
    value=DEFAULT_S0, step=1.0,
)
mu = st.sidebar.slider(
    "Annual Drift μ (%)", -50.0, 100.0,
    DEFAULT_MU * 100, 0.5,
) / 100.0
sigma = st.sidebar.slider(
    "Annual Volatility σ (%)", 0.1, 200.0,
    DEFAULT_SIGMA * 100, 0.5,
) / 100.0
T = st.sidebar.slider(
    "Time Horizon T (years)", 0.1, 5.0, DEFAULT_T, 0.1,
)
steps = st.sidebar.slider(
    "Time Steps", 50, 252, DEFAULT_STEPS, 1,
)
n_paths = st.sidebar.slider(
    "Number of Simulated Paths", 100, MAX_PATHS, DEFAULT_N_PATHS, 100,
)
seed = st.sidebar.number_input(
    "Random Seed", min_value=0, value=DEFAULT_SEED, step=1,
)
rf = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0,
    value=DEFAULT_RF * 100, step=0.1,
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** GBM / Itô process")
st.sidebar.markdown("**Method:** Monte Carlo (exact discrete solution)")

# ----------------------------------------------------------------------
# Simulate (Optimized with TTL Cache to prevent memory overload)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def run_simulation(S0, mu, sigma, T, steps, n_paths, seed):
    return simulate_gbm(S0, mu, sigma, T, steps, n_paths, seed)

with st.spinner("Running Monte Carlo simulation..."):
    time_grid, paths = run_simulation(S0, mu, sigma, T, steps, n_paths, seed)

terminal = paths[:, -1]

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
metrics = compute_risk_metrics(terminal, S0, mu, sigma, T, rf)
analytic_mean = analytical_mean(S0, mu, T)
analytic_std = analytical_std(S0, mu, sigma, T)

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Expected Price E[S_T]",
    f"{metrics['mean']:.2f}",
    f"{(metrics['mean']/S0 - 1) * 100:+.1f}% vs S₀",
)
c2.metric("Median Price", f"{metrics['median']:.2f}")
c3.metric("Std. Deviation", f"{metrics['std']:.2f}")
c4.metric("Probability of Profit", f"{metrics['prob_profit'] * 100:.1f}%")
c5.metric("Volatility Adj. Yield", f"{metrics['vol_adj_yield']:.3f}")

st.markdown("---")

# ----------------------------------------------------------------------
# Chart 1: price paths
# ----------------------------------------------------------------------
st.subheader("Simulated Price Paths")

# Show only a maximum of 100 paths on the graph to keep browser light
max_display = min(MAX_DISPLAYED_PATHS, 100, n_paths)
fig_paths = go.Figure()

for i in range(max_display):
    fig_paths.add_trace(go.Scatter(
        x=time_grid, y=paths[i],
        mode="lines",
        line=dict(width=0.6, color="rgba(0,120,255,0.15)"),
        showlegend=False,
        hoverinfo="skip",
    ))

# Mean / median / quantiles
fig_paths.add_trace(go.Scatter(
    x=time_grid, y=paths.mean(axis=0),
    mode="lines", name="Mean Path",
    line=dict(width=3, color="orange"),
))
fig_paths.add_trace(go.Scatter(
    x=time_grid, y=np.median(paths, axis=0),
    mode="lines", name="Median Path",
    line=dict(width=2, color="green", dash="dash"),
))
fig_paths.add_trace(go.Scatter(
    x=time_grid, y=np.quantile(paths, 0.95, axis=0),
    mode="lines", name="95% Quantile",
    line=dict(width=1.5, color="red"),
))
fig_paths.add_trace(go.Scatter(
    x=time_grid, y=np.quantile(paths, 0.05, axis=0),
    mode="lines", name="5% Quantile",
    line=dict(width=1.5, color="red"),
))

fig_paths.update_layout(
    xaxis_title="Time (years)",
    yaxis_title="Price",
    height=520,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_paths, use_container_width=True)
st.caption(f"Showing {max_display} of {n_paths} simulated paths (Optimized for performance).")

# ----------------------------------------------------------------------
# Chart 2: terminal distribution
# ----------------------------------------------------------------------
st.subheader("Distribution of Terminal Prices")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=terminal, nbinsx=80,
    marker_color="rgba(0,120,255,0.7)",
    name="Terminal prices",
))
fig_hist.add_vline(x=S0, line_dash="dash", line_color="black",
                   annotation_text="S₀", annotation_position="top")
fig_hist.add_vline(x=metrics["q05"], line_dash="dot", line_color="red",
                   annotation_text="5% VaR", annotation_position="top right")
fig_hist.add_vline(x=np.quantile(terminal, 0.95),
                   line_dash="dot", line_color="green",
                   annotation_text="95% quantile", annotation_position="top left")
fig_hist.update_layout(
    xaxis_title="Terminal Price",
    yaxis_title="Frequency",
    height=420,
    showlegend=False,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------------------------
# Risk table
# ----------------------------------------------------------------------
st.subheader("Risk Metrics")

risk_df = pd.DataFrame({
    "Metric": [
        "Expected Price E[S_T] (Monte Carlo)",
        "Analytical E[S_T] = S₀·e^(μT)",
        "Analytical Std. Dev.",
        "Median Price",
        "5% Quantile (Q05)",
        "1% Quantile (Q01)",
        "Value at Risk 95% (% of S₀)",
        "Value at Risk 99% (% of S₀)",
        "Expected Shortfall / CVaR 95%",
        "Probability of Profit",
        "Volatility Adjusted Yield (Define Your Risk™)",
    ],
    "Value": [
        f"{metrics['mean']:.4f}",
        f"{analytic_mean:.4f}",
        f"{analytic_std:.4f}",
        f"{metrics['median']:.4f}",
        f"{metrics['q05']:.4f}",
        f"{metrics['q01']:.4f}",
        f"{metrics['var95'] * 100:.2f}%",
        f"{metrics['var99'] * 100:.2f}%",
        f"{metrics['cvar95'] * 100:.2f}%",
        f"{metrics['prob_profit'] * 100:.2f}%",
        f"{metrics['vol_adj_yield']:.4f}",
    ],
})
st.dataframe(risk_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
with st.expander("🔍 Model Validation (Monte Carlo vs Analytical)"):
    st.write(
        f"- MC E[S_T] = **{metrics['mean']:.4f}**\n"
        f"- Analytical E[S_T] = **{analytic_mean:.4f}**\n"
        f"- Error = **{abs(metrics['mean'] - analytic_mean):.4f}** "
        f"(decreases as paths → ∞)\n\n"
        f"- MC Std(S_T) = **{metrics['std']:.4f}**\n"
        f"- Analytical Std(S_T) = **{analytic_std:.4f}**"
    )

# ----------------------------------------------------------------------
# Education
# ----------------------------------------------------------------------
with st.expander("📚 About the model"):
    st.markdown(
        r"""
**Geometric Brownian Motion (GBM)** is the standard model for stock prices:

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

Using **Itô's Lemma**, the exact solution is:

$$S_t = S_0 \exp\!\Big[(\mu - \tfrac{1}{2}\sigma^2)t + \sigma W_t\Big]$$

Discrete simulation:

$$S_{t+\Delta t} = S_t \exp\!\Big[(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\Big], \quad Z \sim N(0,1)$$

**Volatility Adjusted Yield** (our *Define Your Risk™* metric):

$$\text{VAY} = \frac{(\mu - \tfrac{1}{2}\sigma^2)T - r_f T}{\sigma\sqrt{T}}$$
        """
    )
    st.info(
        "⚠️ GBM assumes constant drift/volatility, no jumps, log-normal returns. "
        "Real markets have fat tails — extend with Heston or jump-diffusion models "
        "for production use."
    )

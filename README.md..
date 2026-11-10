# 📈 Stock Price Scenario Simulator

An interactive **Monte Carlo simulator** that generates thousands of
hypothetical stock-price paths using **Geometric Brownian Motion (GBM)**.

Built for investment education, financial research, and portfolio analysis.

## ✨ Features

- Adjustable drift (μ), volatility (σ), horizon (T), steps, and #paths
- Up to **20,000** simulated paths per run
- Risk metrics: VaR 95/99, CVaR, Probability of Profit
- **Volatility Adjusted Yield** — *Define Your Risk™* metric
- Monte Carlo vs analytical validation
- Interactive Plotly charts

## 🧮 Math

The model is the Itô process:

$$dS = \mu S\,dt + \sigma S\,dW$$

Exact discrete solution:

$$S_{t+\Delta t} = S_t \exp\!\left[(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]$$

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py

"""
Global configuration for the app.
"""

APP_TITLE = "Stock Price Scenario Simulator"
APP_ICON = "📈"
APP_CAPTION = (
    "Monte Carlo simulation of future price paths "
    "using Geometric Brownian Motion"
)

# Default simulation values
DEFAULT_S0 = 100.0
DEFAULT_MU = 0.08        # 8% annual drift
DEFAULT_SIGMA = 0.25     # 25% annual volatility
DEFAULT_T = 1.0          # 1 year
DEFAULT_STEPS = 100      # 100 steps is enough and light on CPU
DEFAULT_N_PATHS = 2000   # 2000 paths is statistically strong but light
DEFAULT_SEED = 42
DEFAULT_RF = 0.04        # 4% risk-free rate

# UI limits
MAX_PATHS = 5000         # Cap at 5000 to prevent platform throttling
MAX_DISPLAYED_PATHS = 100 # Show only 100 paths on the chart

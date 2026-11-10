"""
Risk metrics derived from the terminal price distribution.

Includes:
- Value at Risk (VaR) at 95% and 99%
- Conditional VaR / Expected Shortfall (CVaR)
- Probability of Profit
- Volatility Adjusted Yield (Define Your Risk(TM) metric)
"""

import numpy as np


def compute_risk_metrics(terminal_prices: np.ndarray,
                         S0: float,
                         mu: float,
                         sigma: float,
                         T: float,
                         rf: float) -> dict:
    """
    Compute risk statistics from terminal (end-of-horizon) prices.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of terminal prices from Monte Carlo.
    S0 : float
        Initial price.
    mu : float
        Annual drift.
    sigma : float
        Annual volatility.
    T : float
        Time horizon in years.
    rf : float
        Risk-free rate.

    Returns
    -------
    dict of metrics.
    """
    terminal = np.asarray(terminal_prices)

    mean_price = float(np.mean(terminal))
    median_price = float(np.median(terminal))
    std_price = float(np.std(terminal))

    # Price quantiles
    q05 = float(np.quantile(terminal, 0.05))
    q01 = float(np.quantile(terminal, 0.01))

    # VaR: potential loss as % of initial price
    var_95 = max(0.0, (S0 - q05) / S0)
    var_99 = max(0.0, (S0 - q01) / S0)

    # CVaR / Expected Shortfall: average of worst 5% losses
    worst_5 = terminal[terminal <= q05]
    cvar_95 = max(0.0, (S0 - float(np.mean(worst_5))) / S0)

    # Probability that terminal price exceeds initial price
    prob_profit = float(np.mean(terminal > S0))

    # Volatility Adjusted Yield (Sharpe-like on log-return basis)
    expected_log_return = (mu - 0.5 * sigma ** 2) * T
    total_vol = sigma * np.sqrt(T)
    vol_adj_yield = (expected_log_return - rf * T) / total_vol if total_vol > 0 else 0.0

    return {
        "mean": mean_price,
        "median": median_price,
        "std": std_price,
        "q05": q05,
        "q01": q01,
        "var95": var_95,
        "var99": var_99,
        "cvar95": cvar_95,
        "prob_profit": prob_profit,
        "vol_adj_yield": vol_adj_yield,
    }

"""
Geometric Brownian Motion (GBM) simulation engine.

Model (Itô process):
    dS = mu * S * dt + sigma * S * dW

Exact discrete solution:
    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1)
"""

import numpy as np


def simulate_gbm(S0: float,
                 mu: float,
                 sigma: float,
                 T: float,
                 steps: int,
                 n_paths: int,
                 seed: int = 42):
    """
    Simulate GBM price paths.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Annual drift (e.g. 0.08 for 8%).
    sigma : float
        Annual volatility (e.g. 0.25 for 25%).
    T : float
        Time horizon in years.
    steps : int
        Number of time steps in [0, T].
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    time_grid : np.ndarray, shape (steps+1,)
        Time points from 0 to T.
    paths : np.ndarray, shape (n_paths, steps+1)
        Simulated price paths.
    """
    rng = np.random.default_rng(int(seed))
    dt = T / steps

    # Standard normal shocks: shape (n_paths, steps)
    Z = rng.standard_normal((n_paths, steps))

    # Log-return increments under GBM
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative log-returns, prepend 0 for time 0
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    # Convert log-returns to prices
    paths = S0 * np.exp(log_paths)
    time_grid = np.linspace(0.0, T, steps + 1)

    return time_grid, paths


def analytical_mean(S0: float, mu: float, T: float) -> float:
    """
    Analytical expectation of S_T under GBM: E[S_T] = S0 * exp(mu*T).
    Used for validating Monte Carlo estimates.
    """
    return S0 * np.exp(mu * T)


def analytical_std(S0: float, mu: float, sigma: float, T: float) -> float:
    """
    Analytical standard deviation of S_T under GBM.
    Std = S0 * exp(mu*T) * sqrt(exp(sigma^2*T) - 1)
    """
    return S0 * np.exp(mu * T) * np.sqrt(np.exp(sigma ** 2 * T) - 1)

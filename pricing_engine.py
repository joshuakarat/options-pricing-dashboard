"""Core pricing and visualization functions for a simple derivatives engine.

This module focuses on European options under the Black-Scholes model and a
vectorized Monte Carlo simulator under Geometric Brownian Motion (GBM).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _normalise_option_type(option_type: str) -> OptionType:
    """Validate the option type and convert it to lowercase."""
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    return option_type


def _coerce_arrays(*values: float | np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert all numeric inputs into NumPy arrays for easy broadcasting."""
    return tuple(np.asarray(value, dtype=float) for value in values)


def _coerce_scalar(value: float | np.ndarray, name: str) -> float:
    """Convert a scalar-like input to a Python float."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar for Monte Carlo routines.")
    return float(array)


def _to_output(value: float | np.ndarray) -> float | np.ndarray:
    """Return floats for scalar outputs and arrays for vector outputs."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def _validate_model_inputs(
    S0: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    *,
    allow_zero_maturity: bool,
) -> None:
    """Run lightweight validation on common model inputs."""
    if np.any(S0 <= 0.0):
        raise ValueError("S0 must be strictly positive.")
    if np.any(K <= 0.0):
        raise ValueError("K must be strictly positive.")
    if allow_zero_maturity:
        if np.any(T < 0.0):
            raise ValueError("T cannot be negative.")
    elif np.any(T <= 0.0):
        raise ValueError("T must be strictly positive.")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive.")


def _d1_d2(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Black-Scholes d1 and d2 terms.

    The standard definitions are

    d1 = [ln(S0 / K) + (r - q + 0.5 sigma^2) T] / [sigma sqrt(T)]
    d2 = d1 - sigma sqrt(T)

    ``q`` is a continuous dividend yield. It defaults to zero, which recovers
    the classic non-dividend Black-Scholes formula.
    """
    S0, K, T, r, sigma, q = _coerce_arrays(S0, K, T, r, sigma, q)
    _validate_model_inputs(S0, K, T, sigma, allow_zero_maturity=False)

    variance_term = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / variance_term
    d2 = d1 - variance_term
    return d1, d2


def black_scholes_price(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Price a European call or put using the Black-Scholes formula.

    The Black-Scholes model solves the PDE

    dV/dt + 0.5 sigma^2 S^2 d^2V/dS^2 + r S dV/dS - r V = 0

    with the European terminal payoff as the boundary condition. The resulting
    closed-form prices are

    Call: C = S0 e^(-qT) N(d1) - K e^(-rT) N(d2)
    Put:  P = K e^(-rT) N(-d2) - S0 e^(-qT) N(-d1)

    Parameters
    ----------
    S0 : float or np.ndarray
        Current stock price.
    K : float or np.ndarray
        Strike price.
    T : float or np.ndarray
        Time to maturity in years.
    r : float or np.ndarray
        Continuously compounded risk-free rate.
    sigma : float or np.ndarray
        Volatility of the underlying stock.
    option_type : {"call", "put"}
        Type of European option.
    q : float or np.ndarray, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float or np.ndarray
        The model price, with NumPy broadcasting supported.
    """
    option_type = _normalise_option_type(option_type)
    S0, K, T, r, sigma, q = _coerce_arrays(S0, K, T, r, sigma, q)
    _validate_model_inputs(S0, K, T, sigma, allow_zero_maturity=True)

    intrinsic_value = (
        np.maximum(S0 - K, 0.0)
        if option_type == "call"
        else np.maximum(K - S0, 0.0)
    )

    matured = T == 0.0
    safe_T = np.where(matured, 1.0, T)
    d1, d2 = _d1_d2(S0, K, safe_T, r, sigma, q)

    discount_factor = np.exp(-r * safe_T)
    dividend_discount = np.exp(-q * safe_T)
    if option_type == "call":
        price = S0 * dividend_discount * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
    else:
        price = K * discount_factor * norm.cdf(-d2) - S0 * dividend_discount * norm.cdf(-d1)

    price = np.where(matured, intrinsic_value, price)
    return _to_output(price)


def black_scholes_delta(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute Delta = dV/dS0 for a European option.

    In Black-Scholes:

    Call Delta = e^(-qT) N(d1)
    Put Delta = e^(-qT) [N(d1) - 1]
    """
    option_type = _normalise_option_type(option_type)
    d1, _ = _d1_d2(S0, K, T, r, sigma, q)
    _, _, T, _, _, q = _coerce_arrays(S0, K, T, r, sigma, q)
    dividend_discount = np.exp(-q * T)
    delta = (
        dividend_discount * norm.cdf(d1)
        if option_type == "call"
        else dividend_discount * (norm.cdf(d1) - 1.0)
    )
    return _to_output(delta)


def black_scholes_gamma(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute Gamma = d^2V/dS0^2 for a European option.

    In Black-Scholes:

    Gamma = e^(-qT) phi(d1) / [S0 sigma sqrt(T)]

    where phi is the standard normal probability density function.
    """
    d1, _ = _d1_d2(S0, K, T, r, sigma, q)
    S0, _, T, _, sigma, q = _coerce_arrays(S0, K, T, r, sigma, q)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return _to_output(gamma)


def black_scholes_theta(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute the standard Black-Scholes Theta for a European option.

    Theta measures time decay. With ``T`` written as time to maturity, the
    conventional trader's Theta is the change in option value as calendar time
    passes, so it is typically negative for a long option position.

    Under Black-Scholes:

    Call Theta = -S0 e^(-qT) phi(d1) sigma / [2 sqrt(T)]
                 - r K e^(-rT) N(d2) + q S0 e^(-qT) N(d1)
    Put Theta  = -S0 e^(-qT) phi(d1) sigma / [2 sqrt(T)]
                 + r K e^(-rT) N(-d2) - q S0 e^(-qT) N(-d1)

    The value returned here is in price units per year.
    """
    option_type = _normalise_option_type(option_type)
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    S0, K, T, r, sigma, q = _coerce_arrays(S0, K, T, r, sigma, q)

    dividend_discount = np.exp(-q * T)
    diffusion_term = -S0 * dividend_discount * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    discount_factor = np.exp(-r * T)

    if option_type == "call":
        theta = (
            diffusion_term
            - r * K * discount_factor * norm.cdf(d2)
            + q * S0 * dividend_discount * norm.cdf(d1)
        )
    else:
        theta = (
            diffusion_term
            + r * K * discount_factor * norm.cdf(-d2)
            - q * S0 * dividend_discount * norm.cdf(-d1)
        )
    return _to_output(theta)


def black_scholes_vega(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute Vega = dV/dsigma for a European option.

    In Black-Scholes:

    Vega = S0 e^(-qT) phi(d1) sqrt(T)

    The value returned here is per 1.00 change in volatility, not per 1%.
    """
    d1, _ = _d1_d2(S0, K, T, r, sigma, q)
    S0, _, T, _, _, q = _coerce_arrays(S0, K, T, r, sigma, q)
    vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return _to_output(vega)


def black_scholes_rho(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute Rho = dV/dr for a European option.

    In Black-Scholes:

    Call Rho = K T e^(-rT) N(d2)
    Put Rho  = -K T e^(-rT) N(-d2)

    The value returned here is per 1.00 change in the interest rate.
    """
    option_type = _normalise_option_type(option_type)
    _, d2 = _d1_d2(S0, K, T, r, sigma, q)
    _, K, T, r, _ = _coerce_arrays(S0, K, T, r, sigma)

    discount_factor = np.exp(-r * T)
    rho = (
        K * T * discount_factor * norm.cdf(d2)
        if option_type == "call"
        else -K * T * discount_factor * norm.cdf(-d2)
    )
    return _to_output(rho)


def black_scholes_greeks(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float | np.ndarray = 0.0,
) -> dict[str, float | np.ndarray]:
    """Return the main Black-Scholes Greeks in a single dictionary."""
    return {
        "delta": black_scholes_delta(S0, K, T, r, sigma, option_type, q),
        "gamma": black_scholes_gamma(S0, K, T, r, sigma, q),
        "theta": black_scholes_theta(S0, K, T, r, sigma, option_type, q),
        "vega": black_scholes_vega(S0, K, T, r, sigma, q),
        "rho": black_scholes_rho(S0, K, T, r, sigma, option_type, q),
    }


def implied_volatility(
    market_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    lower_vol: float = 1e-6,
    upper_vol: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Infer Black-Scholes implied volatility from a market option price.

    Black-Scholes normally maps model inputs to an option price:

    S0, K, T, r, sigma -> price

    Implied volatility reverses that relationship by solving for ``sigma`` such
    that the Black-Scholes model price matches the observed market price. This
    function uses Brent's root-finding method because the vanilla option price
    is monotonic in volatility for standard inputs.
    """
    option_type = _normalise_option_type(option_type)
    market_price = _coerce_scalar(market_price, "market_price")
    S0 = _coerce_scalar(S0, "S0")
    K = _coerce_scalar(K, "K")
    T = _coerce_scalar(T, "T")
    r = _coerce_scalar(r, "r")
    q = _coerce_scalar(q, "q")

    if market_price <= 0.0:
        raise ValueError("market_price must be strictly positive.")
    if S0 <= 0.0 or K <= 0.0:
        raise ValueError("S0 and K must be strictly positive.")
    if T <= 0.0:
        raise ValueError("T must be strictly positive.")
    if lower_vol <= 0.0 or upper_vol <= lower_vol:
        raise ValueError("Volatility bounds must satisfy 0 < lower_vol < upper_vol.")

    stock_present_value = S0 * np.exp(-q * T)
    strike_present_value = K * np.exp(-r * T)
    if option_type == "call":
        lower_bound = max(stock_present_value - strike_present_value, 0.0)
        upper_bound = stock_present_value
    else:
        lower_bound = max(strike_present_value - stock_present_value, 0.0)
        upper_bound = strike_present_value

    # A tiny tolerance avoids rejecting prices that differ only by data rounding.
    price_tolerance = 1e-10
    if market_price < lower_bound - price_tolerance:
        raise ValueError("market_price is below the no-arbitrage lower bound.")
    if market_price > upper_bound + price_tolerance:
        raise ValueError("market_price is above the no-arbitrage upper bound.")

    def pricing_error(volatility: float) -> float:
        return (
            black_scholes_price(S0, K, T, r, volatility, option_type, q)
            - market_price
        )

    low_error = pricing_error(lower_vol)
    high_error = pricing_error(upper_vol)
    if abs(low_error) < tolerance:
        return lower_vol
    if abs(high_error) < tolerance:
        return upper_vol
    if low_error * high_error > 0.0:
        raise ValueError("Could not bracket an implied volatility solution.")

    return float(
        brentq(
            pricing_error,
            lower_vol,
            upper_vol,
            xtol=tolerance,
            maxiter=max_iterations,
        )
    )


def monte_carlo_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
    q: float = 0.0,
) -> tuple[float, float]:
    """Price a European option with a vectorized Monte Carlo simulation.

    Under risk-neutral GBM, the terminal stock price satisfies

    S_T = S0 * exp[(r - 0.5 sigma^2) T + sigma sqrt(T) Z]

    where Z ~ N(0, 1). For a European option we only need the terminal price,
    so all paths can be simulated at once with a single vectorized draw.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Standard Black-Scholes inputs.
    option_type : {"call", "put"}
        Type of European option.
    n_paths : int, default 100_000
        Number of Monte Carlo paths.
    seed : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Monte Carlo price estimate and its standard error.
    """
    option_type = _normalise_option_type(option_type)
    S0 = _coerce_scalar(S0, "S0")
    K = _coerce_scalar(K, "K")
    T = _coerce_scalar(T, "T")
    r = _coerce_scalar(r, "r")
    sigma = _coerce_scalar(sigma, "sigma")
    q = _coerce_scalar(q, "q")

    if S0 <= 0.0 or K <= 0.0:
        raise ValueError("S0 and K must be strictly positive.")
    if T <= 0.0 or sigma <= 0.0:
        raise ValueError("T and sigma must be strictly positive.")
    if n_paths <= 1:
        raise ValueError("n_paths must be greater than 1.")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)

    # This is the exact GBM terminal distribution under the risk-neutral measure.
    terminal_prices = S0 * np.exp(
        (r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * shocks
    )

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    price_estimate = float(np.mean(discounted_payoffs))
    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    )
    return price_estimate, standard_error


def monte_carlo_convergence(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    max_paths: int = 200_000,
    checkpoints: np.ndarray | None = None,
    seed: int | None = None,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Track Monte Carlo convergence towards the Black-Scholes price.

    The idea is simple:

    1. Simulate a large block of discounted payoffs.
    2. Compute the running mean with ``np.cumsum``.
    3. Read off the estimate at selected sample sizes.

    This keeps the whole calculation vectorized and makes the convergence to
    the analytic benchmark very easy to visualize.
    """
    option_type = _normalise_option_type(option_type)
    S0 = _coerce_scalar(S0, "S0")
    K = _coerce_scalar(K, "K")
    T = _coerce_scalar(T, "T")
    r = _coerce_scalar(r, "r")
    sigma = _coerce_scalar(sigma, "sigma")
    q = _coerce_scalar(q, "q")

    if max_paths <= 1:
        raise ValueError("max_paths must be greater than 1.")

    if checkpoints is None:
        start = min(100, max_paths)
        checkpoints = np.unique(np.geomspace(start, max_paths, num=30).astype(int))
    else:
        checkpoints = np.unique(np.asarray(checkpoints, dtype=int))

    if checkpoints.size == 0:
        raise ValueError("checkpoints must contain at least one sample size.")
    if checkpoints[0] < 1 or checkpoints[-1] > max_paths:
        raise ValueError("All checkpoints must lie between 1 and max_paths.")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(max_paths)
    terminal_prices = S0 * np.exp(
        (r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * shocks
    )

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    running_estimates = np.cumsum(discounted_payoffs) / np.arange(1, max_paths + 1)
    black_scholes_target = float(
        black_scholes_price(S0, K, T, r, sigma, option_type, q)
    )
    return checkpoints, running_estimates[checkpoints - 1], black_scholes_target


def plot_monte_carlo_convergence(
    checkpoints: np.ndarray,
    estimates: np.ndarray,
    black_scholes_target: float,
    option_type: OptionType = "call",
):
    """Plot the convergence of Monte Carlo estimates to the analytic price."""
    import matplotlib.pyplot as plt

    option_type = _normalise_option_type(option_type)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(checkpoints, estimates, linewidth=2.0, label="Monte Carlo estimate")
    ax.axhline(
        black_scholes_target,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Black-Scholes benchmark",
    )
    ax.set_title(f"Monte Carlo Convergence for a European {option_type.title()}")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Option price")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


def plot_option_price_surface(
    S0: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
):
    """Plot a 3D option price surface against strike and expiry.

    Even though this is sometimes loosely called a "volatility surface" in
    casual conversation, this function plots model price on the vertical axis.
    """
    import matplotlib.pyplot as plt

    option_type = _normalise_option_type(option_type)
    strike_grid, expiry_grid = np.meshgrid(
        np.asarray(strikes, dtype=float),
        np.asarray(expiries, dtype=float),
    )
    price_surface = black_scholes_price(
        S0=S0,
        K=strike_grid,
        T=expiry_grid,
        r=r,
        sigma=sigma,
        option_type=option_type,
        q=q,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        strike_grid,
        expiry_grid,
        price_surface,
        cmap="viridis",
        edgecolor="none",
        alpha=0.95,
    )
    ax.set_title(f"European {option_type.title()} Price Surface")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry (years)")
    ax.set_zlabel("Option price")
    fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.1, label="Option price")
    return fig, ax


def plot_greek_vs_spot(
    spot_prices: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    greek: str = "delta",
    option_type: OptionType = "call",
    q: float = 0.0,
):
    """Plot a chosen Greek against the underlying spot price.

    This is a helpful way to build intuition for how option sensitivities change
    as the option moves in or out of the money.
    """
    import matplotlib.pyplot as plt

    option_type = _normalise_option_type(option_type)
    greek = greek.lower()

    greek_functions = {
        "delta": lambda S: black_scholes_delta(S, K, T, r, sigma, option_type, q),
        "gamma": lambda S: black_scholes_gamma(S, K, T, r, sigma, q),
        "theta": lambda S: black_scholes_theta(S, K, T, r, sigma, option_type, q),
        "vega": lambda S: black_scholes_vega(S, K, T, r, sigma, q),
        "rho": lambda S: black_scholes_rho(S, K, T, r, sigma, option_type, q),
    }
    if greek not in greek_functions:
        raise ValueError(
            "greek must be one of: delta, gamma, theta, vega, rho."
        )

    spot_prices = np.asarray(spot_prices, dtype=float)
    greek_values = greek_functions[greek](spot_prices)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spot_prices, greek_values, linewidth=2.0)
    ax.axvline(K, color="black", linestyle="--", linewidth=1.2, label="Strike")
    ax.set_title(
        f"{greek.title()} vs Spot for a European {option_type.title()}"
    )
    ax.set_xlabel("Spot price")
    ax.set_ylabel(greek.title())
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


__all__ = [
    "black_scholes_price",
    "black_scholes_delta",
    "black_scholes_gamma",
    "black_scholes_theta",
    "black_scholes_vega",
    "black_scholes_rho",
    "black_scholes_greeks",
    "implied_volatility",
    "monte_carlo_european_option",
    "monte_carlo_convergence",
    "plot_monte_carlo_convergence",
    "plot_option_price_surface",
    "plot_greek_vs_spot",
]

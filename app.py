"""Streamlit front end for the options pricing dashboard.

The app is deliberately organised around the project story:

1. Use Black-Scholes and Monte Carlo to understand the model.
2. Pull a real option chain for a listed stock.
3. Invert market prices into implied volatility.
4. Explain the theory clearly enough to defend in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import (
    StockSnapshot,
    fetch_option_chain,
    fetch_option_expiries,
    fetch_price_history,
    fetch_stock_snapshot,
    prepare_option_chain_for_iv,
    realised_volatility,
    rolling_realised_volatility,
)
from pricing_engine import (
    black_scholes_greeks,
    black_scholes_price,
    monte_carlo_convergence,
    monte_carlo_european_option,
)

OptionType = Literal["call", "put"]

THEME = {
    "ink": "#122033",
    "muted": "#667085",
    "paper": "#F6F0E6",
    "panel": "rgba(255, 255, 255, 0.82)",
    "line": "#E2D7C7",
    "teal": "#0E5965",
    "teal_light": "#D8EEF0",
    "copper": "#B96A35",
    "gold": "#DDAA4B",
    "green": "#237A57",
    "red": "#A6423A",
}

CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

GREEKS = ["delta", "gamma", "theta", "vega", "rho"]


@dataclass(frozen=True)
class ModelInputs:
    """Container for the manual model assumptions controlled in the sidebar."""

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    q: float
    option_type: OptionType
    greek: str
    n_paths: int
    seed: int


@dataclass(frozen=True)
class MarketInputs:
    """Container for live market-data controls."""

    ticker: str
    expiry: str | None
    option_type: OptionType
    moneyness_min: float
    moneyness_max: float
    load_requested: bool


def inject_styles() -> None:
    """Apply a professional visual style to the Streamlit app."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(14, 89, 101, 0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(221, 170, 75, 0.18), transparent 30%),
                    linear-gradient(180deg, #F8F2E8 0%, #FBF8F2 42%, #F2E7D8 100%);
                color: {THEME["ink"]};
            }}
            [data-testid="stHeader"] {{
                background: rgba(0, 0, 0, 0);
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #143047 0%, #0E5965 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }}
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] .stMarkdown {{
                color: #FFF8EA !important;
            }}
            [data-testid="stSidebar"] [data-testid="stExpander"] details {{
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 18px;
                overflow: hidden;
            }}
            [data-testid="stSidebar"] [data-testid="stExpander"] summary,
            [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
            [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
                color: {THEME["ink"]} !important;
                font-weight: 800;
            }}
            [data-testid="stSidebar"] [data-testid="stExpander"] label,
            [data-testid="stSidebar"] [data-testid="stExpander"] label p,
            [data-testid="stSidebar"] [data-testid="stExpander"] label span {{
                color: {THEME["ink"]} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stCaptionContainer"] p {{
                color: {THEME["muted"]} !important;
            }}
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] [data-baseweb="input"] input {{
                color: {THEME["ink"]} !important;
                -webkit-text-fill-color: {THEME["ink"]} !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="input"],
            [data-testid="stSidebar"] [data-baseweb="base-input"],
            [data-testid="stSidebar"] [data-baseweb="select"] > div {{
                background: rgba(255, 255, 255, 0.97) !important;
                border-radius: 14px;
                color: {THEME["ink"]} !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] span,
            [data-testid="stSidebar"] [data-baseweb="select"] div {{
                color: {THEME["ink"]} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stSlider"],
            [data-testid="stSidebar"] [data-testid="stSlider"] div,
            [data-testid="stSidebar"] [data-testid="stSlider"] span,
            [data-testid="stSidebar"] [data-testid="stSlider"] p {{
                color: {THEME["ink"]} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
                background-color: {THEME["teal"]} !important;
                border-color: {THEME["teal"]} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBar"] {{
                color: {THEME["muted"]} !important;
            }}
            [data-testid="stSidebar"] button {{
                background: rgba(255, 255, 255, 0.96) !important;
                color: {THEME["ink"]} !important;
                border: 1px solid rgba(18, 32, 51, 0.16) !important;
                border-radius: 14px !important;
            }}
            [data-testid="stSidebar"] button p,
            [data-testid="stSidebar"] button span {{
                color: {THEME["ink"]} !important;
                font-weight: 700;
            }}
            .hero-card {{
                position: relative;
                overflow: hidden;
                padding: 2.1rem 2.3rem;
                border-radius: 30px;
                background: linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.88) 0%,
                    rgba(246, 236, 218, 0.90) 100%
                );
                border: 1px solid rgba(18, 32, 51, 0.08);
                box-shadow: 0 24px 70px rgba(18, 32, 51, 0.09);
                margin-bottom: 1rem;
            }}
            .hero-card::after {{
                content: "";
                position: absolute;
                width: 260px;
                height: 260px;
                top: -95px;
                right: -80px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(185, 106, 53, 0.28), transparent 70%);
            }}
            .eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                font-weight: 800;
                color: {THEME["teal"]};
                margin-bottom: 0.65rem;
            }}
            .hero-title {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: clamp(2.25rem, 5vw, 3.8rem);
                line-height: 1.03;
                color: {THEME["ink"]};
                margin: 0 0 0.65rem 0;
            }}
            .hero-copy {{
                max-width: 64rem;
                font-size: 1.03rem;
                line-height: 1.75;
                color: {THEME["muted"]};
                margin: 0;
            }}
            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin: 0.7rem 0 1rem 0;
            }}
            .pill {{
                border-radius: 999px;
                padding: 0.56rem 0.9rem;
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid rgba(18, 32, 51, 0.08);
                color: {THEME["ink"]};
                font-size: 0.92rem;
                box-shadow: 0 10px 22px rgba(18, 32, 51, 0.05);
            }}
            .pill strong {{
                color: {THEME["teal"]};
            }}
            .section-card {{
                padding: 1.15rem 1.2rem;
                border-radius: 24px;
                background: {THEME["panel"]};
                border: 1px solid {THEME["line"]};
                box-shadow: 0 18px 42px rgba(18, 32, 51, 0.06);
                margin-bottom: 1rem;
            }}
            .section-title {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.32rem;
                color: {THEME["ink"]};
                margin: 0 0 0.35rem 0;
            }}
            .section-copy {{
                font-size: 0.96rem;
                line-height: 1.7;
                color: {THEME["muted"]};
                margin: 0;
            }}
            .metric-card {{
                min-height: 132px;
                padding: 1rem 1.1rem;
                border-radius: 24px;
                background: {THEME["panel"]};
                border: 1px solid {THEME["line"]};
                box-shadow: 0 16px 36px rgba(18, 32, 51, 0.06);
            }}
            .metric-label {{
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.72rem;
                font-weight: 800;
                color: {THEME["muted"]};
                margin-bottom: 0.55rem;
            }}
            .metric-value {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: clamp(1.5rem, 3vw, 2.15rem);
                line-height: 1.05;
                color: {THEME["ink"]};
                margin-bottom: 0.45rem;
            }}
            .metric-footnote {{
                font-size: 0.88rem;
                line-height: 1.5;
                color: {THEME["muted"]};
            }}
            .sidebar-note {{
                padding: 0.92rem 1rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                color: #FFF8EA;
                font-size: 0.92rem;
                line-height: 1.55;
                margin: 0.35rem 0 0.8rem 0;
            }}
            .formula-card {{
                padding: 1.05rem 1.15rem;
                border-radius: 22px;
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid {THEME["line"]};
                box-shadow: 0 12px 28px rgba(18, 32, 51, 0.05);
                margin-bottom: 1rem;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.45rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 999px;
                padding: 0.65rem 1rem;
                background: rgba(255, 255, 255, 0.60);
                border: 1px solid rgba(18, 32, 51, 0.08);
            }}
            .stTabs [aria-selected="true"] {{
                background: {THEME["teal"]} !important;
                color: white !important;
            }}
            div[data-testid="stDataFrame"] {{
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid {THEME["line"]};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_money(value: float, currency: str = "$") -> str:
    """Format a numeric value as money."""
    if not np.isfinite(value):
        return "n/a"
    return f"{currency}{value:,.2f}"


def format_pct(value: float) -> str:
    """Format a decimal as a percentage."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def format_number(value: float, decimals: int = 4) -> str:
    """Format a generic numeric value with a fixed number of decimals."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def render_metric_card(label: str, value: str, footnote: str) -> None:
    """Render a consistent custom metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_figure_layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    """Apply common Plotly styling."""
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.68)",
        font={"family": "Georgia, serif", "color": THEME["ink"]},
        margin={"l": 28, "r": 28, "t": 62, "b": 34},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
    )
    fig.update_xaxes(
        gridcolor="rgba(18,32,51,0.10)",
        zerolinecolor="rgba(18,32,51,0.20)",
    )
    fig.update_yaxes(
        gridcolor="rgba(18,32,51,0.10)",
        zerolinecolor="rgba(18,32,51,0.20)",
    )
    return fig


@st.cache_data(ttl=900, show_spinner=False)
def cached_expiries(ticker: str) -> list[str]:
    """Cached wrapper around the live expiry fetch."""
    return fetch_option_expiries(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_snapshot(ticker: str) -> StockSnapshot:
    """Cached wrapper around the live stock snapshot fetch."""
    return fetch_stock_snapshot(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Cached wrapper around the live option-chain fetch."""
    return fetch_option_chain(ticker, expiry)


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(ticker: str) -> pd.DataFrame:
    """Cached wrapper around the live price-history fetch."""
    return fetch_price_history(ticker, period="1y")


def render_header() -> None:
    """Render the dashboard hero section."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Derivative Pricing Engine</div>
            <h1 class="hero-title">Options Pricing Dashboard</h1>
            <p class="hero-copy">
                A compact quant finance project that moves from theory to
                market calibration: Black-Scholes pricing, Greeks, vectorized
                Monte Carlo, live option chains, and implied-volatility smiles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pill-row">
            <div class="pill"><strong>Model</strong> Black-Scholes + Greeks</div>
            <div class="pill"><strong>Simulation</strong> Vectorized GBM Monte Carlo</div>
            <div class="pill"><strong>Market</strong> Listed option chains</div>
            <div class="pill"><strong>Calibration</strong> Implied volatility smile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar() -> tuple[ModelInputs, MarketInputs]:
    """Collect all user-controlled assumptions from the sidebar."""
    st.sidebar.markdown("## Assumptions")
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            Use the first section for clean model experiments. Use the market
            section when you want a real stock option chain and IV smile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Model Pricer", expanded=True):
        S0 = st.number_input("Spot price (S0)", min_value=0.01, value=100.0, step=1.0)
        K = st.number_input("Strike price (K)", min_value=0.01, value=100.0, step=1.0)
        T = st.number_input("Time to expiry (years)", min_value=0.01, value=1.0, step=0.05)
        r_pct = st.number_input("Risk-free rate (%)", value=5.0, step=0.25)
        sigma_pct = st.number_input("Volatility sigma (%)", min_value=0.01, value=20.0, step=1.0)
        q_pct = st.number_input("Dividend yield q (%)", min_value=0.0, value=0.0, step=0.25)
        option_type = st.selectbox("Option type", ["call", "put"], index=0)
        greek = st.selectbox("Greek profile", GREEKS, index=0)
        n_paths = st.slider(
            "Monte Carlo paths",
            min_value=10_000,
            max_value=500_000,
            value=150_000,
            step=10_000,
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    with st.sidebar.expander("Market Chain", expanded=True):
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        market_option_type = st.selectbox("Market option type", ["call", "put"], index=0)

        expiry_options: list[str] = []
        expiry_error = ""
        if ticker:
            try:
                expiry_options = cached_expiries(ticker)
            except Exception as exc:
                expiry_error = str(exc)

        if expiry_options:
            default_index = min(2, len(expiry_options) - 1)
            expiry = st.selectbox("Expiry", expiry_options, index=default_index)
        else:
            expiry = st.text_input("Expiry (YYYY-MM-DD)", value="")
            if expiry_error:
                st.caption("Expiry list unavailable. You can still type an expiry manually.")

        moneyness_range = st.slider(
            "Smile strike range (K / S0)",
            min_value=0.50,
            max_value=1.50,
            value=(0.80, 1.20),
            step=0.01,
        )
        load_requested = st.button("Load / refresh market chain", use_container_width=True)

    model_inputs = ModelInputs(
        S0=S0,
        K=K,
        T=T,
        r=r_pct / 100.0,
        sigma=sigma_pct / 100.0,
        q=q_pct / 100.0,
        option_type=option_type,  # type: ignore[arg-type]
        greek=greek,
        n_paths=int(n_paths),
        seed=int(seed),
    )
    market_inputs = MarketInputs(
        ticker=ticker,
        expiry=expiry or None,
        option_type=market_option_type,  # type: ignore[arg-type]
        moneyness_min=moneyness_range[0],
        moneyness_max=moneyness_range[1],
        load_requested=load_requested,
    )
    return model_inputs, market_inputs


def build_convergence_figure(inputs: ModelInputs) -> tuple[go.Figure, float, float, float]:
    """Create the Monte Carlo convergence chart."""
    mc_price, mc_error = monte_carlo_european_option(
        S0=inputs.S0,
        K=inputs.K,
        T=inputs.T,
        r=inputs.r,
        sigma=inputs.sigma,
        option_type=inputs.option_type,
        n_paths=inputs.n_paths,
        seed=inputs.seed,
        q=inputs.q,
    )
    checkpoints = np.unique(
        np.geomspace(min(250, inputs.n_paths), inputs.n_paths, num=34).astype(int)
    )
    checkpoints, estimates, benchmark = monte_carlo_convergence(
        S0=inputs.S0,
        K=inputs.K,
        T=inputs.T,
        r=inputs.r,
        sigma=inputs.sigma,
        option_type=inputs.option_type,
        max_paths=inputs.n_paths,
        checkpoints=checkpoints,
        seed=inputs.seed,
        q=inputs.q,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=estimates,
            mode="lines",
            name="Monte Carlo estimate",
            line={"color": THEME["copper"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=np.full_like(estimates, benchmark, dtype=float),
            mode="lines",
            name="Black-Scholes target",
            line={"color": THEME["ink"], "width": 2, "dash": "dash"},
        )
    )
    fig.update_xaxes(title="Number of simulated paths", type="log")
    fig.update_yaxes(title="Option price")
    base_figure_layout(fig, "Monte Carlo Convergence", height=420)
    return fig, mc_price, mc_error, benchmark


def build_price_surface_figure(inputs: ModelInputs) -> go.Figure:
    """Create a 3D model price surface over strike and expiry."""
    strikes = np.linspace(max(inputs.S0 * 0.55, 1.0), inputs.S0 * 1.45, 52)
    expiries = np.linspace(0.05, max(2.0, inputs.T * 1.5), 52)
    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
    price_surface = black_scholes_price(
        S0=inputs.S0,
        K=strike_grid,
        T=expiry_grid,
        r=inputs.r,
        sigma=inputs.sigma,
        option_type=inputs.option_type,
        q=inputs.q,
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=strike_grid,
                y=expiry_grid,
                z=price_surface,
                colorscale="Tealgrn",
                showscale=True,
                colorbar={"title": "Price"},
            )
        ]
    )
    fig.update_layout(
        scene={
            "xaxis_title": "Strike K",
            "yaxis_title": "Expiry T",
            "zaxis_title": "Option price",
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 0.95}},
        }
    )
    base_figure_layout(fig, "Black-Scholes Price Surface", height=560)
    return fig


def build_greek_profile_figure(inputs: ModelInputs) -> go.Figure:
    """Create a Greek profile as spot price changes."""
    spot_grid = np.linspace(max(inputs.S0 * 0.50, 1.0), inputs.S0 * 1.50, 260)
    greek_values = black_scholes_greeks(
        S0=spot_grid,
        K=inputs.K,
        T=inputs.T,
        r=inputs.r,
        sigma=inputs.sigma,
        option_type=inputs.option_type,
        q=inputs.q,
    )[inputs.greek]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spot_grid,
            y=greek_values,
            mode="lines",
            name=inputs.greek.title(),
            line={"color": THEME["teal"], "width": 3},
        )
    )
    fig.add_vline(
        x=inputs.K,
        line_dash="dash",
        line_color=THEME["ink"],
        annotation_text="Strike",
    )
    fig.update_xaxes(title="Underlying spot price S")
    fig.update_yaxes(title=inputs.greek.title())
    base_figure_layout(fig, f"{inputs.greek.title()} vs Spot", height=420)
    return fig


def build_greek_comparison_table(inputs: ModelInputs) -> pd.DataFrame:
    """Return call and put Greeks for the active model assumptions."""
    call_greeks = black_scholes_greeks(
        inputs.S0, inputs.K, inputs.T, inputs.r, inputs.sigma, "call", inputs.q
    )
    put_greeks = black_scholes_greeks(
        inputs.S0, inputs.K, inputs.T, inputs.r, inputs.sigma, "put", inputs.q
    )
    rows = []
    for greek in GREEKS:
        rows.append(
            {
                "Greek": greek.title(),
                "Call": float(call_greeks[greek]),
                "Put": float(put_greeks[greek]),
            }
        )
    return pd.DataFrame(rows)


def render_model_pricer(inputs: ModelInputs) -> None:
    """Render the manual model-pricing tab."""
    call_price = float(
        black_scholes_price(
            inputs.S0, inputs.K, inputs.T, inputs.r, inputs.sigma, "call", inputs.q
        )
    )
    put_price = float(
        black_scholes_price(
            inputs.S0, inputs.K, inputs.T, inputs.r, inputs.sigma, "put", inputs.q
        )
    )
    convergence_fig, mc_price, mc_error, benchmark = build_convergence_figure(inputs)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Model Pricer</div>
            <p class="section-copy">
                This tab is the clean theoretical engine. The closed-form
                Black-Scholes price is the benchmark; Monte Carlo simulates the
                same risk-neutral terminal stock distribution and should
                converge towards that benchmark as paths increase.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Call Price", format_money(call_price), "Closed-form Black-Scholes")
    with col2:
        render_metric_card("Put Price", format_money(put_price), "Closed-form Black-Scholes")
    with col3:
        render_metric_card(
            "Monte Carlo",
            format_money(mc_price),
            f"Std. error {mc_error:.4f}",
        )
    with col4:
        render_metric_card(
            "MC Difference",
            format_money(abs(mc_price - benchmark)),
            "Absolute gap to analytic target",
        )

    left, right = st.columns([1.08, 0.92])
    with left:
        st.plotly_chart(convergence_fig, use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Call and Put Greek Comparison</div>
                <p class="section-copy">
                    Greeks are partial derivatives of the option value. They
                    explain how the model price reacts when one input changes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        greek_table = build_greek_comparison_table(inputs)
        st.dataframe(
            greek_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Call": st.column_config.NumberColumn(format="%.6f"),
                "Put": st.column_config.NumberColumn(format="%.6f"),
            },
        )

    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.plotly_chart(
            build_price_surface_figure(inputs),
            use_container_width=True,
            config=CHART_CONFIG,
        )
    with chart_right:
        st.plotly_chart(
            build_greek_profile_figure(inputs),
            use_container_width=True,
            config=CHART_CONFIG,
        )


def load_market_bundle(
    market_inputs: MarketInputs,
    r: float,
    fallback_q: float,
) -> dict[str, object]:
    """Fetch live market data and prepare the IV table."""
    if not market_inputs.expiry:
        raise ValueError("Choose or type an expiry before loading market data.")

    snapshot = cached_snapshot(market_inputs.ticker)
    dividend_yield = snapshot.dividend_yield if snapshot.dividend_yield > 0 else fallback_q
    option_chain = cached_chain(market_inputs.ticker, market_inputs.expiry)
    history = cached_history(market_inputs.ticker)
    prepared_chain = prepare_option_chain_for_iv(
        option_chain=option_chain,
        S0=snapshot.spot,
        expiry=market_inputs.expiry,
        r=r,
        option_type=market_inputs.option_type,
        q=dividend_yield,
    )
    realised_vol = realised_volatility(history["Close"])
    rolling_vol = rolling_realised_volatility(history["Close"])

    return {
        "snapshot": snapshot,
        "option_chain": option_chain,
        "prepared_chain": prepared_chain,
        "history": history,
        "rolling_vol": rolling_vol,
        "realised_vol": realised_vol,
        "r": r,
        "q": dividend_yield,
        "expiry": market_inputs.expiry,
        "option_type": market_inputs.option_type,
        "ticker": market_inputs.ticker,
    }


def maybe_load_market_data(market_inputs: MarketInputs, model_inputs: ModelInputs) -> None:
    """Load market data when the sidebar button is pressed."""
    if not market_inputs.load_requested:
        return
    with st.spinner("Fetching option chain and solving implied volatility..."):
        try:
            st.session_state["market_bundle"] = load_market_bundle(
                market_inputs=market_inputs,
                r=model_inputs.r,
                fallback_q=model_inputs.q,
            )
            st.session_state["market_error"] = ""
        except Exception as exc:
            st.session_state["market_error"] = str(exc)


def get_market_bundle() -> dict[str, object] | None:
    """Return the latest loaded market bundle from session state."""
    bundle = st.session_state.get("market_bundle")
    if isinstance(bundle, dict):
        return bundle
    return None


def filter_prepared_chain(
    prepared_chain: pd.DataFrame,
    market_inputs: MarketInputs,
) -> pd.DataFrame:
    """Apply the active moneyness filter to a prepared IV table."""
    if prepared_chain.empty:
        return prepared_chain
    return prepared_chain[
        prepared_chain["moneyness"].between(
            market_inputs.moneyness_min,
            market_inputs.moneyness_max,
        )
    ].reset_index(drop=True)


def render_market_prompt() -> None:
    """Explain how to load live market data when none has been fetched yet."""
    st.info(
        "Choose a ticker and expiry in the sidebar, then press "
        "`Load / refresh market chain`. The model tab works without live data; "
        "the market tabs need internet access and the yfinance dependency."
    )
    error = st.session_state.get("market_error")
    if error:
        st.warning(f"Market data could not be loaded: {error}")


def build_history_figure(history: pd.DataFrame, rolling_vol: pd.Series) -> go.Figure:
    """Plot stock price and rolling realised volatility."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Adjusted close",
            line={"color": THEME["teal"], "width": 2.5},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode="lines",
            name="30D realised volatility",
            line={"color": THEME["copper"], "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis={"title": "Stock price"},
        yaxis2={
            "title": "Annualised vol",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
        },
    )
    base_figure_layout(fig, "Underlying Price and Realised Volatility", height=430)
    return fig


def render_market_chain(market_inputs: MarketInputs) -> None:
    """Render the live market-chain tab."""
    bundle = get_market_bundle()
    if bundle is None:
        render_market_prompt()
        return

    snapshot = bundle["snapshot"]
    prepared_chain = bundle["prepared_chain"]
    history = bundle["history"]
    rolling_vol = bundle["rolling_vol"]
    realised_vol = float(bundle["realised_vol"])
    q = float(bundle["q"])
    expiry = str(bundle["expiry"])
    option_type = str(bundle["option_type"])

    assert isinstance(snapshot, StockSnapshot)
    assert isinstance(prepared_chain, pd.DataFrame)
    assert isinstance(history, pd.DataFrame)
    assert isinstance(rolling_vol, pd.Series)

    filtered_chain = filter_prepared_chain(prepared_chain, market_inputs)

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Market Chain</div>
            <p class="section-copy">
                Live listed options for <strong>{snapshot.ticker}</strong>.
                The chain is cleaned by preferring bid/ask mid prices, then
                falling back to last traded prices when needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Underlying", snapshot.ticker, snapshot.name)
    with col2:
        render_metric_card("Spot Price", format_money(snapshot.spot), snapshot.currency)
    with col3:
        render_metric_card("Realised Vol", format_pct(realised_vol), "1Y daily log returns")
    with col4:
        render_metric_card("Dividend Yield", format_pct(q), "Used as continuous q")

    st.plotly_chart(
        build_history_figure(history, rolling_vol),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{option_type.title()} Chain for {expiry}</div>
            <p class="section-copy">
                Showing strikes where K / S0 is between
                {market_inputs.moneyness_min:.2f} and {market_inputs.moneyness_max:.2f}.
                The computed IV column is solved by the project code, not copied
                from the provider.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_chain.empty:
        st.warning("No contracts survived the current filters. Widen the moneyness range.")
        return

    display_columns = [
        "contractSymbol",
        "strike",
        "moneyness",
        "bid",
        "ask",
        "lastPrice",
        "marketPrice",
        "priceSource",
        "volume",
        "openInterest",
        "modelIV",
        "providerIV",
    ]
    st.dataframe(
        filtered_chain[display_columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "moneyness": st.column_config.NumberColumn(format="%.3f"),
            "bid": st.column_config.NumberColumn(format="$%.2f"),
            "ask": st.column_config.NumberColumn(format="$%.2f"),
            "lastPrice": st.column_config.NumberColumn(format="$%.2f"),
            "marketPrice": st.column_config.NumberColumn(format="$%.2f"),
            "modelIV": st.column_config.NumberColumn("Computed IV", format="%.2%"),
            "providerIV": st.column_config.NumberColumn("Provider IV", format="%.2%"),
            "volume": st.column_config.NumberColumn(format="%.0f"),
            "openInterest": st.column_config.NumberColumn(format="%.0f"),
        },
    )


def build_iv_smile_figure(
    filtered_chain: pd.DataFrame,
    spot: float,
    option_type: str,
) -> go.Figure:
    """Build the implied-volatility smile chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_chain["strike"],
            y=filtered_chain["modelIV"],
            mode="markers+lines",
            name="Computed IV",
            marker={
                "size": 8,
                "color": filtered_chain["moneyness"],
                "colorscale": "Tealgrn",
                "showscale": True,
                "colorbar": {"title": "K / S0"},
            },
            line={"color": THEME["teal"], "width": 2},
            customdata=np.stack(
                [
                    filtered_chain["moneyness"],
                    filtered_chain["marketPrice"],
                    filtered_chain["priceSource"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Strike: %{x:.2f}<br>"
                "IV: %{y:.2%}<br>"
                "K / S0: %{customdata[0]:.3f}<br>"
                "Market price: $%{customdata[1]:.2f}<br>"
                "Price source: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    if "providerIV" in filtered_chain and filtered_chain["providerIV"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=filtered_chain["strike"],
                y=filtered_chain["providerIV"],
                mode="markers",
                name="Provider IV",
                marker={"size": 7, "color": THEME["gold"], "symbol": "diamond"},
                hovertemplate="Strike: %{x:.2f}<br>Provider IV: %{y:.2%}<extra></extra>",
            )
        )

    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color=THEME["ink"],
        annotation_text="Spot",
    )
    fig.update_xaxes(title="Strike K")
    fig.update_yaxes(title="Implied volatility", tickformat=".0%")
    base_figure_layout(fig, f"{option_type.title()} Implied Volatility Smile", height=520)
    return fig


def build_market_vs_model_figure(filtered_chain: pd.DataFrame) -> go.Figure:
    """Show that the solved IV reprices the market mid by construction."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_chain["strike"],
            y=filtered_chain["marketPrice"],
            mode="markers+lines",
            name="Market price",
            marker={"color": THEME["copper"], "size": 8},
            line={"color": THEME["copper"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_chain["strike"],
            y=filtered_chain["modelPriceAtIV"],
            mode="lines",
            name="BS price using computed IV",
            line={"color": THEME["teal"], "width": 2.5, "dash": "dash"},
        )
    )
    fig.update_xaxes(title="Strike K")
    fig.update_yaxes(title="Option price")
    base_figure_layout(fig, "Market Price Repricing Check", height=390)
    return fig


def render_iv_smile(market_inputs: MarketInputs) -> None:
    """Render the implied-volatility smile tab."""
    bundle = get_market_bundle()
    if bundle is None:
        render_market_prompt()
        return

    snapshot = bundle["snapshot"]
    prepared_chain = bundle["prepared_chain"]
    option_type = str(bundle["option_type"])
    expiry = str(bundle["expiry"])

    assert isinstance(snapshot, StockSnapshot)
    assert isinstance(prepared_chain, pd.DataFrame)

    filtered_chain = filter_prepared_chain(prepared_chain, market_inputs)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Implied Volatility Smile</div>
            <p class="section-copy">
                Implied volatility asks: what volatility would make
                Black-Scholes reproduce the observed market option price? A
                flat line would match the simplest Black-Scholes assumption;
                real markets usually show a smile or skew.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_chain.empty:
        st.warning("No contracts survived the current filters. Widen the moneyness range.")
        return

    atm_index = (filtered_chain["moneyness"] - 1.0).abs().idxmin()
    atm_row = filtered_chain.loc[atm_index]
    left, middle, right = st.columns(3)
    with left:
        render_metric_card("Contracts", f"{len(filtered_chain):,}", "After current filters")
    with middle:
        render_metric_card("ATM-ish IV", format_pct(float(atm_row["modelIV"])), f"Strike {atm_row['strike']:.2f}")
    with right:
        render_metric_card("Expiry", expiry, f"{option_type.title()} options")

    st.plotly_chart(
        build_iv_smile_figure(filtered_chain, snapshot.spot, option_type),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    st.plotly_chart(
        build_market_vs_model_figure(filtered_chain),
        use_container_width=True,
        config=CHART_CONFIG,
    )


def render_theory(model_inputs: ModelInputs) -> None:
    """Render the methodology and interview-explanation tab."""
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Theory and Methodology</div>
            <p class="section-copy">
                This tab is designed to help you explain the project. The key
                idea is that each layer answers a different quant question:
                theoretical price, simulation validation, market data, then
                market-implied volatility.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Black-Scholes PDE")
    st.markdown(
        r"""
        For option value \(V(S,t)\), the Black-Scholes PDE is

        $$
        \frac{\partial V}{\partial t}
        + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
        + (r-q)S\frac{\partial V}{\partial S}
        - rV = 0.
        $$

        Here \(S\) is the stock price, \(r\) is the risk-free rate,
        \(\sigma\) is volatility, and \(q\) is a continuous dividend yield.
        For European calls and puts this PDE has a closed-form solution.
        """
    )

    st.markdown("### 2. Greeks")
    st.markdown(
        """
        Greeks are the local sensitivities of the option price. Delta measures
        sensitivity to spot, Gamma measures curvature, Theta measures time
        decay, Vega measures sensitivity to volatility, and Rho measures
        sensitivity to interest rates. In the code these are closed-form
        derivatives, so they are fast and deterministic.
        """
    )

    st.markdown("### 3. Monte Carlo")
    st.markdown(
        r"""
        Under the risk-neutral measure, the terminal GBM stock price is

        $$
        S_T = S_0
        \exp\left((r-q-\tfrac{1}{2}\sigma^2)T
        + \sigma\sqrt{T}Z\right),
        \qquad Z \sim \mathcal{N}(0,1).
        $$

        For a European option only \(S_T\) is needed. The implementation draws
        all random shocks at once with NumPy, computes every payoff in one
        vectorized operation, discounts the average payoff, and reports a
        standard error.
        """
    )

    st.markdown("### 4. Market Calibration and IV Smile")
    st.markdown(
        """
        Market option prices rarely line up with one constant volatility. The
        dashboard therefore takes each listed contract, chooses a market price
        from the bid/ask midpoint where possible, and solves for the implied
        volatility that makes Black-Scholes match that price. Plotting IV
        against strike gives the volatility smile or skew.
        """
    )

    st.markdown("### Active Model Assumptions")
    assumptions = pd.DataFrame(
        [
            {"Input": "Spot S0", "Value": f"{model_inputs.S0:.2f}"},
            {"Input": "Strike K", "Value": f"{model_inputs.K:.2f}"},
            {"Input": "Maturity T", "Value": f"{model_inputs.T:.2f} years"},
            {"Input": "Risk-free rate r", "Value": format_pct(model_inputs.r)},
            {"Input": "Dividend yield q", "Value": format_pct(model_inputs.q)},
            {"Input": "Volatility sigma", "Value": format_pct(model_inputs.sigma)},
            {"Input": "Option type", "Value": model_inputs.option_type.title()},
        ]
    )
    st.dataframe(assumptions, hide_index=True, use_container_width=True)

    st.caption(
        "Educational note: yfinance data is convenient for portfolio projects, "
        "but not a replacement for professional market data feeds."
    )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Options Pricing Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    model_inputs, market_inputs = build_sidebar()
    maybe_load_market_data(market_inputs, model_inputs)

    render_header()
    tab_model, tab_market, tab_smile, tab_theory = st.tabs(
        ["Model Pricer", "Market Chain", "IV Smile", "Theory"]
    )

    with tab_model:
        render_model_pricer(model_inputs)
    with tab_market:
        render_market_chain(market_inputs)
    with tab_smile:
        render_iv_smile(market_inputs)
    with tab_theory:
        render_theory(model_inputs)


if __name__ == "__main__":
    main()

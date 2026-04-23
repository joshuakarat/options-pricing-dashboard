"""Market-data helpers for option-chain calibration.

The pricing engine is intentionally pure mathematics. This module handles the
messier real-world layer: fetching listed option chains, choosing a usable
market price, estimating historical volatility, and inverting option prices
into implied volatility.

The data-fetching functions use ``yfinance`` because it is simple and free for
educational projects. Market data quality varies, so the cleaning functions are
kept explicit and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from pricing_engine import black_scholes_price, implied_volatility

OptionType = Literal["call", "put"]

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StockSnapshot:
    """Small summary of the underlying stock used by the dashboard."""

    ticker: str
    spot: float
    name: str
    currency: str
    dividend_yield: float


def _load_yfinance() -> Any:
    """Import yfinance only when live market data is requested.

    Keeping this import lazy means the core project and tests still run even if
    the optional market-data dependency has not been installed yet.
    """
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live market data requires yfinance. Install it with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return yf


def _as_float(value: Any, default: float = np.nan) -> float:
    """Convert a market-data field to float while tolerating missing values."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(converted):
        return default
    return converted


def _normalise_ticker(ticker: str) -> str:
    """Clean ticker input before sending it to the data provider."""
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker cannot be blank.")
    return cleaned


def _normalise_dividend_yield(value: Any) -> float:
    """Convert dividend yield into decimal form, e.g. 1.5% -> 0.015."""
    dividend_yield = _as_float(value, default=0.0)
    if dividend_yield < 0.0:
        return 0.0
    if dividend_yield > 1.0:
        return dividend_yield / 100.0
    return dividend_yield


def year_fraction_to_expiry(expiry: str, today: date | None = None) -> float:
    """Convert an option expiry date into Black-Scholes time ``T`` in years.

    Black-Scholes uses time to maturity as a year fraction. Listed option chains
    usually give expiry as a date, so this function bridges the market-data
    convention and the model convention.
    """
    today = today or date.today()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - today).days
    return max(days_to_expiry, 1) / 365.0


def fetch_stock_snapshot(ticker: str) -> StockSnapshot:
    """Fetch the latest available stock price and basic metadata.

    The latest close is used as the spot price if an intraday last price is not
    available. This is sufficient for an educational option-pricing dashboard,
    but professional trading systems would use a licensed real-time feed.
    """
    yf = _load_yfinance()
    ticker = _normalise_ticker(ticker)
    ticker_object = yf.Ticker(ticker)

    info: dict[str, Any]
    try:
        info = ticker_object.info or {}
    except Exception:
        info = {}

    spot = np.nan
    for spot_field in ("regularMarketPrice", "currentPrice", "previousClose"):
        spot = _as_float(info.get(spot_field))
        if np.isfinite(spot):
            break

    if not np.isfinite(spot):
        try:
            history = ticker_object.history(period="5d", auto_adjust=False)
        except Exception as exc:
            raise RuntimeError(f"Could not fetch a valid spot price for {ticker}.") from exc
        if history.empty or "Close" not in history:
            raise RuntimeError(f"Could not fetch a valid spot price for {ticker}.")
        spot = float(history["Close"].dropna().iloc[-1])

    name = str(info.get("longName") or info.get("shortName") or ticker)
    currency = str(info.get("currency") or "USD")
    dividend_yield = _normalise_dividend_yield(info.get("dividendYield"))

    return StockSnapshot(
        ticker=ticker,
        spot=spot,
        name=name,
        currency=currency,
        dividend_yield=dividend_yield,
    )


def fetch_option_expiries(ticker: str) -> list[str]:
    """Return available listed option expiries for a ticker."""
    yf = _load_yfinance()
    ticker = _normalise_ticker(ticker)
    return list(yf.Ticker(ticker).options)


def fetch_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Fetch and combine call and put chains for one expiry date."""
    yf = _load_yfinance()
    ticker = _normalise_ticker(ticker)
    chain = yf.Ticker(ticker).option_chain(expiry)

    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["optionType"] = "call"
    puts["optionType"] = "put"

    option_chain = pd.concat([calls, puts], ignore_index=True)
    if option_chain.empty:
        raise RuntimeError(f"No option-chain rows returned for {ticker} {expiry}.")
    return option_chain


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical stock prices for realised-volatility estimates."""
    yf = _load_yfinance()
    ticker = _normalise_ticker(ticker)
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise RuntimeError(f"Could not fetch price history for {ticker}.")
    return history


def realised_volatility(
    close_prices: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Estimate annualised historical volatility from log returns.

    Daily log returns are used because they add cleanly over time. The standard
    deviation is annualised by multiplying by ``sqrt(252)``, the usual
    approximation for trading days in a year.
    """
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_returns.empty:
        return np.nan
    return float(log_returns.std(ddof=1) * np.sqrt(trading_days))


def rolling_realised_volatility(
    close_prices: pd.Series,
    window: int = 30,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute rolling annualised volatility from daily log returns."""
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def select_market_price(row: pd.Series) -> tuple[float, str]:
    """Choose the market price used for implied-volatility inversion.

    The bid/ask midpoint is preferred because it approximates the fair tradable
    level better than the last traded price. If the quote is missing or crossed,
    the function falls back to ``lastPrice``.
    """
    bid = _as_float(row.get("bid"))
    ask = _as_float(row.get("ask"))
    last_price = _as_float(row.get("lastPrice"))

    if bid > 0.0 and ask > 0.0 and ask >= bid:
        return 0.5 * (bid + ask), "mid"
    if last_price > 0.0:
        return last_price, "last"
    return np.nan, "missing"


def prepare_option_chain_for_iv(
    option_chain: pd.DataFrame,
    S0: float,
    expiry: str,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    today: date | None = None,
) -> pd.DataFrame:
    """Clean an option chain and solve implied volatility row by row.

    Each listed contract gives a market price for a specific strike and expiry.
    Implied volatility is the volatility that makes the Black-Scholes price
    equal that market price. Plotting implied volatility against strike reveals
    the market's volatility smile/skew.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    if "strike" not in option_chain:
        raise ValueError("option_chain must contain a 'strike' column.")

    T = year_fraction_to_expiry(expiry, today=today)
    filtered_chain = option_chain.copy()
    if "optionType" in filtered_chain:
        filtered_chain = filtered_chain[
            filtered_chain["optionType"].astype(str).str.lower() == option_type
        ]

    records: list[dict[str, float | str]] = []
    for _, row in filtered_chain.iterrows():
        strike = _as_float(row.get("strike"))
        market_price, price_source = select_market_price(row)
        if strike <= 0.0 or market_price <= 0.0:
            continue

        try:
            model_iv = implied_volatility(
                market_price=market_price,
                S0=S0,
                K=strike,
                T=T,
                r=r,
                option_type=option_type,
                q=q,
            )
        except ValueError:
            continue

        model_price = black_scholes_price(
            S0=S0,
            K=strike,
            T=T,
            r=r,
            sigma=model_iv,
            option_type=option_type,
            q=q,
        )
        bid = _as_float(row.get("bid"))
        ask = _as_float(row.get("ask"))
        spread = ask - bid if ask >= bid and bid > 0.0 else np.nan

        records.append(
            {
                "contractSymbol": str(row.get("contractSymbol", "")),
                "optionType": option_type,
                "expiry": expiry,
                "T": T,
                "strike": strike,
                "moneyness": strike / S0,
                "bid": bid,
                "ask": ask,
                "lastPrice": _as_float(row.get("lastPrice")),
                "marketPrice": market_price,
                "priceSource": price_source,
                "spread": spread,
                "volume": _as_float(row.get("volume"), default=0.0),
                "openInterest": _as_float(row.get("openInterest"), default=0.0),
                "providerIV": _as_float(row.get("impliedVolatility")),
                "modelIV": model_iv,
                "modelPriceAtIV": float(model_price),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "contractSymbol",
                "optionType",
                "expiry",
                "T",
                "strike",
                "moneyness",
                "bid",
                "ask",
                "lastPrice",
                "marketPrice",
                "priceSource",
                "spread",
                "volume",
                "openInterest",
                "providerIV",
                "modelIV",
                "modelPriceAtIV",
            ]
        )

    prepared = pd.DataFrame(records).sort_values("strike").reset_index(drop=True)
    return prepared

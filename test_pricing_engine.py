"""Tests for the derivative pricing engine."""

from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from market_data import prepare_option_chain_for_iv, select_market_price
from pricing_engine import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    implied_volatility,
    monte_carlo_convergence,
    monte_carlo_european_option,
)


class BlackScholesTests(unittest.TestCase):
    """Regression tests for analytic Black-Scholes outputs."""

    def setUp(self) -> None:
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20

    def test_reference_prices_match_standard_benchmark(self) -> None:
        """Prices should match the textbook at-the-money benchmark."""
        call_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "call"
        )
        put_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "put"
        )

        self.assertAlmostEqual(call_price, 10.4505835722, places=8)
        self.assertAlmostEqual(put_price, 5.5735260223, places=8)

    def test_put_call_parity_holds(self) -> None:
        """European call and put prices should satisfy put-call parity."""
        call_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "call"
        )
        put_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "put"
        )
        parity_gap = call_price - put_price - (
            self.S0 - self.K * np.exp(-self.r * self.T)
        )
        self.assertAlmostEqual(parity_gap, 0.0, places=10)

    def test_put_call_parity_holds_with_dividend_yield(self) -> None:
        """Dividend-adjusted parity should use the discounted stock value."""
        dividend_yield = 0.015
        call_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "call", dividend_yield
        )
        put_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "put", dividend_yield
        )
        stock_present_value = self.S0 * np.exp(-dividend_yield * self.T)
        strike_present_value = self.K * np.exp(-self.r * self.T)
        parity_gap = call_price - put_price - (
            stock_present_value - strike_present_value
        )

        self.assertAlmostEqual(parity_gap, 0.0, places=10)

    def test_implied_volatility_recovers_known_sigma(self) -> None:
        """Inverting a Black-Scholes price should recover the input volatility."""
        for option_type in ("call", "put"):
            market_price = black_scholes_price(
                self.S0, self.K, self.T, self.r, self.sigma, option_type
            )
            recovered_sigma = implied_volatility(
                market_price,
                self.S0,
                self.K,
                self.T,
                self.r,
                option_type=option_type,
            )

            self.assertAlmostEqual(recovered_sigma, self.sigma, places=7)

    def test_vectorized_pricing_broadcasts_correctly(self) -> None:
        """The analytic pricer should support vectorized NumPy inputs."""
        strikes = np.array([90.0, 100.0, 110.0])
        prices = black_scholes_price(
            self.S0, strikes, self.T, self.r, self.sigma, "call"
        )

        self.assertEqual(prices.shape, (3,))
        self.assertTrue(np.all(np.diff(prices) < 0.0))

    def test_reference_delta_and_gamma_are_stable(self) -> None:
        """A small Greek regression test catches accidental formula changes."""
        delta = black_scholes_delta(
            self.S0, self.K, self.T, self.r, self.sigma, "call"
        )
        gamma = black_scholes_gamma(
            self.S0, self.K, self.T, self.r, self.sigma
        )

        self.assertAlmostEqual(delta, 0.6368306512, places=8)
        self.assertAlmostEqual(gamma, 0.0187620173, places=8)


class MonteCarloTests(unittest.TestCase):
    """Tests for the vectorized Monte Carlo implementation."""

    def setUp(self) -> None:
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20

    def test_monte_carlo_matches_black_scholes_within_statistical_noise(self) -> None:
        """The Monte Carlo estimate should be close to the analytic benchmark."""
        analytic_price = black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "call"
        )
        mc_price, mc_error = monte_carlo_european_option(
            self.S0,
            self.K,
            self.T,
            self.r,
            self.sigma,
            option_type="call",
            n_paths=200_000,
            seed=42,
        )

        self.assertLess(abs(mc_price - analytic_price), 4.0 * mc_error)

    def test_convergence_estimate_improves_with_more_paths(self) -> None:
        """Later convergence checkpoints should be closer to the benchmark."""
        checkpoints, estimates, benchmark = monte_carlo_convergence(
            self.S0,
            self.K,
            self.T,
            self.r,
            self.sigma,
            option_type="call",
            max_paths=10_000,
            seed=42,
        )

        self.assertEqual(checkpoints.ndim, 1)
        self.assertEqual(estimates.ndim, 1)
        self.assertEqual(len(checkpoints), len(estimates))
        self.assertLess(abs(estimates[-1] - benchmark), abs(estimates[0] - benchmark))


class MarketDataPreparationTests(unittest.TestCase):
    """Tests for deterministic option-chain cleaning and IV inversion."""

    def test_market_price_prefers_bid_ask_midpoint(self) -> None:
        """The bid/ask midpoint is preferred over last traded price."""
        row = pd.Series({"bid": 9.80, "ask": 10.20, "lastPrice": 11.50})
        price, source = select_market_price(row)

        self.assertEqual(source, "mid")
        self.assertAlmostEqual(price, 10.00)

    def test_market_price_falls_back_to_last_price(self) -> None:
        """The last traded price is used when the quote is unusable."""
        row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 2.75})
        price, source = select_market_price(row)

        self.assertEqual(source, "last")
        self.assertAlmostEqual(price, 2.75)

    def test_prepare_option_chain_solves_computed_iv(self) -> None:
        """A synthetic chain row should recover the known model volatility."""
        market_price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
        option_chain = pd.DataFrame(
            {
                "contractSymbol": ["TEST260101C00100000"],
                "optionType": ["call"],
                "strike": [100.0],
                "bid": [market_price - 0.01],
                "ask": [market_price + 0.01],
                "lastPrice": [market_price],
                "volume": [10],
                "openInterest": [100],
                "impliedVolatility": [0.21],
            }
        )

        prepared = prepare_option_chain_for_iv(
            option_chain=option_chain,
            S0=100.0,
            expiry="2027-01-01",
            r=0.05,
            option_type="call",
            today=date(2026, 1, 1),
        )

        self.assertEqual(len(prepared), 1)
        self.assertAlmostEqual(prepared.loc[0, "modelIV"], 0.20, places=5)


if __name__ == "__main__":
    unittest.main()

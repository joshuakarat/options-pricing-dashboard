# Options Pricing Dashboard

A Python derivatives pricing project that connects option-pricing theory with
live market option chains. The project starts with a readable Black-Scholes and
Monte Carlo pricing engine, then uses listed option prices to compute implied
volatility smiles.

This is designed as a portfolio-quality quant finance project: mathematically
clear, easy to run, tested, and presentation-ready through Streamlit.

## Highlights

- Black-Scholes pricing for European calls and puts
- Closed-form Greeks: Delta, Gamma, Theta, Vega, and Rho
- Continuous dividend yield support through `q`
- Vectorized Monte Carlo pricing under Geometric Brownian Motion
- Monte Carlo convergence against the analytic Black-Scholes benchmark
- Live stock option chains through `yfinance`
- Implied volatility solver using Brent root finding
- IV smile/skew visualization by strike and moneyness
- Interactive four-tab Streamlit dashboard
- Unit tests and GitHub Actions CI

## Dashboard

Run the app and move through the project in four stages:

1. `Model Pricer`: price European options, inspect Greeks, run Monte Carlo, and view model surfaces.
2. `Market Chain`: load a listed stock option chain and inspect real contracts.
3. `IV Smile`: invert market prices into implied volatilities and plot the smile.
4. `Theory`: review the math and methodology behind the engine.

```bash
streamlit run app.py
```

## Mathematical Background

### Black-Scholes PDE

For derivative value \(V(S,t)\), the Black-Scholes PDE with continuous dividend
yield \(q\) is

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
$$

For European options, the closed-form solution uses

$$
d_1 =
\frac{\ln(S_0/K) + \left(r-q+\frac{1}{2}\sigma^2\right)T}
{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}.
$$

The call and put prices are

$$
C = S_0e^{-qT}N(d_1) - Ke^{-rT}N(d_2),
$$

$$
P = Ke^{-rT}N(-d_2) - S_0e^{-qT}N(-d_1).
$$

The code uses `scipy.stats.norm` for the normal CDF \(N(\cdot)\) and PDF
\(\phi(\cdot)\).

### Greeks

The Greeks are closed-form partial derivatives of the option price:

- Delta: sensitivity to the underlying stock price
- Gamma: curvature with respect to the stock price
- Theta: time decay
- Vega: sensitivity to volatility
- Rho: sensitivity to the interest rate

They are implemented directly in `pricing_engine.py` so the formulas remain
visible and easy to explain.

### Monte Carlo Simulation

Under the risk-neutral measure, the terminal stock price follows

$$
S_T = S_0 \exp\left[\left(r - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right],
\qquad Z \sim \mathcal{N}(0,1).
$$

For a European option, only \(S_T\) is needed. The simulation therefore draws
all normal shocks with NumPy in one vectorized operation, computes all payoffs
at once, discounts the mean payoff, and reports a standard error.

### Implied Volatility

Black-Scholes maps volatility to price:

```text
sigma -> model price
```

Implied volatility reverses that relationship:

```text
market price -> sigma implied by the market
```

The project solves for the volatility that makes the Black-Scholes price match
the observed option-chain price. Plotting that implied volatility against
strike creates the IV smile or skew.

## Project Structure

```text
.
|-- .github/workflows/ci.yml   # Runs tests automatically on GitHub
|-- .streamlit/config.toml     # Streamlit theme configuration
|-- app.py                     # Four-tab interactive dashboard
|-- market_data.py             # Option-chain fetching, cleaning, and IV prep
|-- pricing_engine.py          # Black-Scholes, Greeks, IV solver, Monte Carlo
|-- main.py                    # Command-line demo and figure generator
|-- tests/
|   `-- test_pricing_engine.py # Unit tests for pricing and market prep
|-- figures/                   # Generated plots and screenshots
|-- requirements.txt           # Python dependencies
|-- pyproject.toml             # Formatter/linter settings
|-- LICENSE
`-- README.md
```

## Installation

Clone the repository, create a virtual environment, and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Running the App

```bash
streamlit run app.py
```

The model tab works offline. The market-chain and IV-smile tabs need internet
access because they fetch option-chain data through `yfinance`.

## Running the CLI Demo

The default command generates example figures into `figures/`:

```bash
python3 main.py
```

View all configurable inputs with:

```bash
python3 main.py --help
```

Example custom run:

```bash
python3 main.py \
  --spot 120 \
  --strike 110 \
  --maturity 0.5 \
  --rate 0.03 \
  --dividend-yield 0.01 \
  --volatility 0.25 \
  --option-type put \
  --greek gamma \
  --n-paths 300000 \
  --file-prefix custom_case
```

## Testing

Run the unit tests with:

```bash
python3 -m unittest discover -s tests -v
```

The tests check:

- benchmark Black-Scholes prices
- put-call parity with and without dividends
- vectorized NumPy broadcasting
- stable Delta and Gamma reference values
- Monte Carlo agreement with the analytic benchmark
- implied volatility recovery from known model prices
- option-chain midpoint selection and IV preparation

## Notes on Market Data

This project uses `yfinance` because it is free and convenient for educational
use. For professional trading or production risk systems, a licensed market
data feed would be required. The dashboard should therefore be treated as a
learning and portfolio project, not a trading system.

## License

MIT License. See `LICENSE` for details.

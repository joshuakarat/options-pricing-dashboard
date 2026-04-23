"""Command-line demo for the derivative pricing engine.

Run ``python3 main.py --help`` to see all available market and plotting inputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep Matplotlib cache files inside the project so the script works cleanly
# in sandboxed or fresh environments.
MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for main.py. Install dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc

# The Agg backend lets us save figures even in headless environments.
matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc

import numpy as np

from pricing_engine import (
    black_scholes_greeks,
    black_scholes_price,
    monte_carlo_convergence,
    monte_carlo_european_option,
    plot_greek_vs_spot,
    plot_monte_carlo_convergence,
    plot_option_price_surface,
)

DEFAULTS = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.00,
    "option_type": "call",
    "greek": "delta",
    "n_paths": 200_000,
    "seed": 42,
    "surface_strike_min": 60.0,
    "surface_strike_max": 140.0,
    "surface_expiry_min": 0.10,
    "surface_expiry_max": 2.00,
    "surface_points": 45,
    "spot_min": 50.0,
    "spot_max": 150.0,
    "spot_points": 300,
    "output_dir": "figures",
    "file_prefix": "",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the demo script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a Black-Scholes and Monte Carlo demonstration for a "
            "European option."
        )
    )
    parser.add_argument("--spot", type=float, default=DEFAULTS["spot"], help="Spot price S0.")
    parser.add_argument("--strike", type=float, default=DEFAULTS["strike"], help="Strike price K.")
    parser.add_argument(
        "--maturity",
        type=float,
        default=DEFAULTS["maturity"],
        help="Time to maturity T in years.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULTS["rate"],
        help="Continuously compounded risk-free rate r.",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=DEFAULTS["volatility"],
        help="Volatility sigma.",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=DEFAULTS["dividend_yield"],
        help="Continuous dividend yield q.",
    )
    parser.add_argument(
        "--option-type",
        choices=["call", "put"],
        default=DEFAULTS["option_type"],
        help="Option type used for Monte Carlo and plots.",
    )
    parser.add_argument(
        "--greek",
        choices=["delta", "gamma", "theta", "vega", "rho"],
        default=DEFAULTS["greek"],
        help="Greek shown in the 2D sensitivity plot.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=DEFAULTS["n_paths"],
        help="Number of Monte Carlo paths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Random seed for reproducible Monte Carlo output.",
    )
    parser.add_argument(
        "--surface-strike-min",
        type=float,
        default=DEFAULTS["surface_strike_min"],
        help="Minimum strike shown on the price surface.",
    )
    parser.add_argument(
        "--surface-strike-max",
        type=float,
        default=DEFAULTS["surface_strike_max"],
        help="Maximum strike shown on the price surface.",
    )
    parser.add_argument(
        "--surface-expiry-min",
        type=float,
        default=DEFAULTS["surface_expiry_min"],
        help="Minimum expiry shown on the price surface.",
    )
    parser.add_argument(
        "--surface-expiry-max",
        type=float,
        default=DEFAULTS["surface_expiry_max"],
        help="Maximum expiry shown on the price surface.",
    )
    parser.add_argument(
        "--surface-points",
        type=int,
        default=DEFAULTS["surface_points"],
        help="Number of grid points used per surface axis.",
    )
    parser.add_argument(
        "--spot-min",
        type=float,
        default=DEFAULTS["spot_min"],
        help="Minimum spot used in the Greek profile plot.",
    )
    parser.add_argument(
        "--spot-max",
        type=float,
        default=DEFAULTS["spot_max"],
        help="Maximum spot used in the Greek profile plot.",
    )
    parser.add_argument(
        "--spot-points",
        type=int,
        default=DEFAULTS["spot_points"],
        help="Number of points used in the Greek profile plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULTS["output_dir"],
        help="Directory where generated figures are saved.",
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULTS["file_prefix"],
        help="Optional prefix added to every saved figure filename.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line inputs before pricing begins."""
    positive_fields = {
        "spot": args.spot,
        "strike": args.strike,
        "maturity": args.maturity,
        "volatility": args.volatility,
        "surface_strike_min": args.surface_strike_min,
        "surface_strike_max": args.surface_strike_max,
        "surface_expiry_min": args.surface_expiry_min,
        "surface_expiry_max": args.surface_expiry_max,
        "spot_min": args.spot_min,
        "spot_max": args.spot_max,
    }
    for name, value in positive_fields.items():
        if value <= 0.0:
            raise SystemExit(f"{name} must be strictly positive.")
    if args.dividend_yield < 0.0:
        raise SystemExit("dividend yield cannot be negative.")

    integer_fields = {
        "n_paths": args.n_paths,
        "surface_points": args.surface_points,
        "spot_points": args.spot_points,
    }
    for name, value in integer_fields.items():
        if value <= 1:
            raise SystemExit(f"{name} must be greater than 1.")

    if args.surface_strike_min >= args.surface_strike_max:
        raise SystemExit("surface strike min must be smaller than surface strike max.")
    if args.surface_expiry_min >= args.surface_expiry_max:
        raise SystemExit("surface expiry min must be smaller than surface expiry max.")
    if args.spot_min >= args.spot_max:
        raise SystemExit("spot min must be smaller than spot max.")


def _print_section(title: str) -> None:
    """Print a simple title block for terminal output."""
    print(f"\n{title}")
    print("-" * len(title))


def _print_greeks(title: str, greeks: dict[str, float | np.ndarray]) -> None:
    """Print a nicely aligned Greek summary."""
    _print_section(title)
    for name, value in greeks.items():
        print(f"{name.title():<6}: {float(value):.6f}")


def _parameter_summary_text(args: argparse.Namespace) -> str:
    """Create a short parameter summary for console output and plots."""
    return (
        f"S0={args.spot:.2f}, K={args.strike:.2f}, T={args.maturity:.2f}, "
        f"r={args.rate:.2%}, q={args.dividend_yield:.2%}, "
        f"sigma={args.volatility:.2%}, "
        f"paths={args.n_paths:,}, seed={args.seed}"
    )


def _attach_parameter_box(fig, text: str) -> None:
    """Add a compact parameter box below a figure."""
    fig.text(
        0.5,
        0.02,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#c8c8c8"},
    )


def _styled_filename(prefix: str, stem: str) -> str:
    """Build a saved filename with an optional user-provided prefix."""
    return f"{prefix}_{stem}" if prefix else stem


def run_demo(args: argparse.Namespace) -> None:
    """Run the pricing demo with user-supplied parameters."""
    plt.style.use("seaborn-v0_8-whitegrid")

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_section("Input Parameters")
    print(_parameter_summary_text(args))
    print(
        f"Plot focus: {args.option_type.title()} option, "
        f"{args.greek.title()} sensitivity"
    )

    call_price = black_scholes_price(
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility,
        "call",
        args.dividend_yield,
    )
    put_price = black_scholes_price(
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility,
        "put",
        args.dividend_yield,
    )

    _print_section("Black-Scholes Prices")
    print(f"European call price: {call_price:.6f}")
    print(f"European put price : {put_price:.6f}")

    call_greeks = black_scholes_greeks(
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility,
        option_type="call",
        q=args.dividend_yield,
    )
    put_greeks = black_scholes_greeks(
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility,
        option_type="put",
        q=args.dividend_yield,
    )
    _print_greeks("Call Greeks", call_greeks)
    _print_greeks("Put Greeks", put_greeks)

    analytic_target = black_scholes_price(
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility,
        option_type=args.option_type,
        q=args.dividend_yield,
    )
    mc_price, mc_error = monte_carlo_european_option(
        S0=args.spot,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        sigma=args.volatility,
        option_type=args.option_type,
        n_paths=args.n_paths,
        seed=args.seed,
        q=args.dividend_yield,
    )

    _print_section("Monte Carlo vs Black-Scholes")
    print(f"Option type              : {args.option_type.title()}")
    print(f"Black-Scholes price      : {analytic_target:.6f}")
    print(f"Monte Carlo price        : {mc_price:.6f}")
    print(f"Monte Carlo std. error   : {mc_error:.6f}")
    print(f"Absolute difference      : {abs(mc_price - analytic_target):.6f}")

    checkpoint_start = min(250, args.n_paths)
    checkpoints = np.unique(
        np.geomspace(checkpoint_start, args.n_paths, num=35).astype(int)
    )
    checkpoints, estimates, benchmark = monte_carlo_convergence(
        S0=args.spot,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        sigma=args.volatility,
        option_type=args.option_type,
        max_paths=args.n_paths,
        checkpoints=checkpoints,
        seed=args.seed,
        q=args.dividend_yield,
    )

    parameter_text = _parameter_summary_text(args)

    convergence_fig, convergence_ax = plot_monte_carlo_convergence(
        checkpoints,
        estimates,
        benchmark,
        option_type=args.option_type,
    )
    convergence_ax.set_title(
        f"Monte Carlo Convergence: {args.option_type.title()} Option"
    )
    _attach_parameter_box(convergence_fig, parameter_text)
    convergence_fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    convergence_path = output_dir / _styled_filename(
        args.file_prefix, "mc_convergence.png"
    )
    convergence_fig.savefig(convergence_path, dpi=220, bbox_inches="tight")
    plt.close(convergence_fig)

    strikes = np.linspace(
        args.surface_strike_min, args.surface_strike_max, args.surface_points
    )
    expiries = np.linspace(
        args.surface_expiry_min, args.surface_expiry_max, args.surface_points
    )
    surface_fig, surface_ax = plot_option_price_surface(
        S0=args.spot,
        strikes=strikes,
        expiries=expiries,
        r=args.rate,
        sigma=args.volatility,
        option_type=args.option_type,
        q=args.dividend_yield,
    )
    surface_ax.view_init(elev=28, azim=-130)
    _attach_parameter_box(surface_fig, parameter_text)
    surface_fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    surface_path = output_dir / _styled_filename(
        args.file_prefix, f"{args.option_type}_price_surface.png"
    )
    surface_fig.savefig(surface_path, dpi=220, bbox_inches="tight")
    plt.close(surface_fig)

    spot_grid = np.linspace(args.spot_min, args.spot_max, args.spot_points)
    greek_fig, greek_ax = plot_greek_vs_spot(
        spot_prices=spot_grid,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        sigma=args.volatility,
        greek=args.greek,
        option_type=args.option_type,
        q=args.dividend_yield,
    )
    greek_ax.set_title(
        f"{args.greek.title()} Profile: {args.option_type.title()} Option"
    )
    _attach_parameter_box(greek_fig, parameter_text)
    greek_fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    greek_path = output_dir / _styled_filename(
        args.file_prefix, f"{args.option_type}_{args.greek}_profile.png"
    )
    greek_fig.savefig(greek_path, dpi=220, bbox_inches="tight")
    plt.close(greek_fig)

    _print_section("Saved Figures")
    print(f"Monte Carlo convergence : {convergence_path}")
    print(f"Price surface           : {surface_path}")
    print(f"Greek profile           : {greek_path}")


def main() -> None:
    """Parse inputs and run the end-to-end demo."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_demo(args)


if __name__ == "__main__":
    main()

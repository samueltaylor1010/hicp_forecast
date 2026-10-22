"""Plot historical HICP indices and a joint-fit shock scenario for one country."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASELINE_FILE = "baselines.csv"
MULTIPLIER_FILE = "multipliers.csv"
HISTORY_FILE = "historical_indices.csv"
DEFAULT_COUNTRY = "FR"
DEFAULT_SHOCK = "CP045"
DEFAULT_SHOCK_SIZE = 0.10
COLORS = {
    "history": "#1f4e79",
    "baseline": "#e67e22",
    "scenario": "#c0392b",
}


def load_artifacts(
    baseline_path: str | Path,
    multiplier_path: str | Path,
    history_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three serialized pipeline artifacts."""
    baselines = pd.read_csv(
        baseline_path,
        parse_dates=["forecast_origin", "target_period"],
    )
    multipliers = pd.read_csv(multiplier_path)
    history = pd.read_csv(history_path, parse_dates=["date"])
    return baselines, multipliers, history


def make_plot(
    baselines: pd.DataFrame,
    multipliers: pd.DataFrame,
    history: pd.DataFrame,
    country: str,
    shock_coicop: str,
    shock_size: float,
    output: Path,
) -> Path:
    """Render a multi-panel historical, baseline, and scenario chart."""
    country_baselines = baselines[baselines["country"] == country]
    country_history = history[history["country"] == country]
    country_multipliers = multipliers[
        (multipliers["country"] == country)
        & (multipliers["shock_variable"] == shock_coicop)
    ]

    if country_baselines.empty:
        raise ValueError(f"No baselines found for {country}")
    if country_history.empty:
        raise ValueError(f"No historical indices found for {country}")
    if country_baselines["target_period"].nunique() != 1:
        raise ValueError(f"Multiple target periods found for {country}")
    if country_multipliers.duplicated("target_coicop").any():
        raise ValueError(f"Duplicate multipliers for {country}/{shock_coicop}")

    target_period = country_baselines["target_period"].iloc[0]
    baseline_map = country_baselines.set_index("coicop")["baseline_index_t3"]
    beta_map = country_multipliers.set_index("target_coicop")["beta_multiplier"]

    history_pivot = country_history.pivot(
        index="date",
        columns="coicop",
        values="index_value",
    ).sort_index()
    components = [shock_coicop] + [
        component
        for component in sorted(baseline_map.index)
        if component != shock_coicop
    ]

    missing_history = set(components) - set(history_pivot.columns)
    if missing_history:
        raise ValueError(
            f"Missing historical series for {country}: {sorted(missing_history)}"
        )

    ncols = 5
    nrows = int(np.ceil(len(components) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, component in zip(axes, components):
        series = history_pivot[component].dropna()
        ax.plot(
            series.index,
            series.to_numpy(),
            color=COLORS["history"],
            linewidth=1.6,
        )

        baseline_index = float(baseline_map.loc[component])
        ax.scatter(
            [target_period],
            [baseline_index],
            color=COLORS["baseline"],
            edgecolor="white",
            linewidth=0.6,
            marker="o",
            s=42,
            zorder=4,
        )

        beta = beta_map.get(component)
        if pd.notna(beta):
            scenario_index = baseline_index * np.exp(shock_size * float(beta))
            scenario_change = 100.0 * np.expm1(shock_size * float(beta))
            ax.scatter(
                [target_period],
                [scenario_index],
                color=COLORS["scenario"],
                edgecolor="white",
                linewidth=0.6,
                marker="*",
                s=90,
                zorder=5,
            )
            ax.text(
                0.03,
                0.06,
                f"scenario vs baseline: {scenario_change:+.2f}%",
                transform=ax.transAxes,
                fontsize=7.5,
                color=COLORS["scenario"],
            )
        else:
            ax.text(
                0.03,
                0.06,
                "AR-only baseline\nno pass-through multiplier",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#555555",
            )

        ax.axvline(
            series.index.max(),
            color="#7f8c8d",
            linestyle=":",
            linewidth=1.0,
        )
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8)
        title = f"{component} (shock source)" if component == shock_coicop else component
        ax.set_title(title, fontsize=10, fontweight="bold")
        if component not in beta_map.index:
            ax.set_facecolor("#fafafa")

    for ax in axes[len(components):]:
        ax.set_visible(False)

    for ax in axes[(nrows - 1) * ncols:nrows * ncols]:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for row in range(nrows):
        axes[row * ncols].set_ylabel("Index")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["history"],
            linewidth=2,
            label="Historical index (last 36 months)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["baseline"],
            marker="o",
            linestyle="None",
            markersize=7,
            label=f"Conditional baseline t+3 ({target_period:%Y-%m})",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["scenario"],
            marker="*",
            linestyle="None",
            markersize=11,
            label=f"Scenario: {shock_coicop} +{shock_size:.0%}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=10,
    )
    fig.suptitle(
        f"{country}: {shock_coicop} joint-fit shock scenario across HICP components",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Baseline is conditional on realized CP045 and CP0722 for multiplier "
        "targets; CP045 and CP0722 use AR-only baselines.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.08,
        hspace=0.38,
        wspace=0.24,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help="Target geography code, for example FR.",
    )
    parser.add_argument(
        "--shock-coicop",
        default=DEFAULT_SHOCK,
        help="Shocked component, default CP045.",
    )
    parser.add_argument(
        "--shock-size",
        type=float,
        default=DEFAULT_SHOCK_SIZE,
        help="Shock in log points; 0.10 means a 10 percent shock.",
    )
    parser.add_argument("--baselines", default=BASELINE_FILE)
    parser.add_argument("--multipliers", default=MULTIPLIER_FILE)
    parser.add_argument("--history", default=HISTORY_FILE)
    parser.add_argument("--output", help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    """Load artifacts and render the requested scenario."""
    args = parse_args()
    if not np.isfinite(args.shock_size):
        raise ValueError("Shock size must be finite")

    country = args.country.upper()
    shock_coicop = args.shock_coicop.upper()
    baselines, multipliers, history = load_artifacts(
        args.baselines,
        args.multipliers,
        args.history,
    )
    output = Path(
        args.output or f"scenario_jointfit_{country}_{shock_coicop}.png"
    )
    path = make_plot(
        baselines,
        multipliers,
        history,
        country,
        shock_coicop,
        args.shock_size,
        output,
    )
    print(path)


if __name__ == "__main__":
    main()

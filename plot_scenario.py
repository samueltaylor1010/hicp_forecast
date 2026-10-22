"""Plot historical HICP indices and a simultaneous two-shock scenario.

CP045 and CP0722 are not pass-through targets, so their scenario values are
derived by applying the origin shock to their own AR lag-0 coefficient.
"""

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
DEFAULT_SHOCK_SIZE = 0.10
SHOCK_COICOPS = ("CP045", "CP0722")
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
    shock_size: float,
    output: Path,
) -> Path:
    """Render historical, baseline, and two-shock scenario panels.

    Multiplier targets use their conditional baseline plus pass-through. The
    shock sources use their AR-only baseline plus their own lag-0 response.
    """
    country_baselines = baselines[baselines["country"] == country]
    country_history = history[history["country"] == country]
    country_multipliers = multipliers[
        (multipliers["country"] == country)
        & (multipliers["shock_variable"].isin(SHOCK_COICOPS))
    ]

    if country_baselines.empty:
        raise ValueError(f"No baselines found for {country}")
    if country_history.empty:
        raise ValueError(f"No historical indices found for {country}")
    if country_baselines["target_period"].nunique() != 1:
        raise ValueError(f"Multiple target periods found for {country}")
    if country_multipliers.duplicated(
        ["target_coicop", "shock_variable"]
    ).any():
        raise ValueError(f"Duplicate multipliers for {country}")

    target_period = country_baselines["target_period"].iloc[0]
    if "lag0_coefficient" not in country_baselines.columns:
        raise ValueError(
            "Baselines are missing lag0_coefficient; rerun hicp_forecast.py"
        )
    baseline_map = country_baselines.set_index("coicop")["baseline_index_t3"]
    lag0_map = country_baselines.set_index("coicop")["lag0_coefficient"]
    beta_table = country_multipliers.pivot(
        index="target_coicop",
        columns="shock_variable",
        values="beta_multiplier",
    )
    if set(beta_table.columns) != set(SHOCK_COICOPS):
        raise ValueError(
            f"Missing shock multipliers for {country}: {sorted(beta_table.columns)}"
        )

    expected_targets = set(baseline_map.index) - set(SHOCK_COICOPS)
    if set(beta_table.index) != expected_targets:
        raise ValueError(f"Unexpected multiplier targets for {country}")

    history_pivot = country_history.pivot(
        index="date",
        columns="coicop",
        values="index_value",
    ).sort_index()
    components = [
        *[component for component in SHOCK_COICOPS if component in baseline_map.index],
        *[
            component
            for component in sorted(baseline_map.index)
            if component not in SHOCK_COICOPS
        ],
    ]

    missing_history = set(components) - set(history_pivot.columns)
    if missing_history:
        raise ValueError(
            f"Missing historical series for {country}: {sorted(missing_history)}"
        )

    for component in SHOCK_COICOPS:
        if component not in lag0_map.index:
            raise ValueError(
                f"Missing lag0 coefficient for {country}/{component}"
            )
        if not np.isfinite(float(lag0_map.loc[component])):
            raise ValueError(
                f"Invalid lag0 coefficient for {country}/{component}"
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

        if component in beta_table.index:
            betas = beta_table.loc[component]
            delta_yoy = shock_size * float(
                betas[SHOCK_COICOPS[0]] + betas[SHOCK_COICOPS[1]]
            )
            if not np.isfinite(delta_yoy):
                raise ValueError(f"Invalid multipliers for {country}/{component}")
            scenario_index = baseline_index * np.exp(delta_yoy)
            scenario_change = 100.0 * np.expm1(delta_yoy)
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
                f"simultaneous scenario vs baseline: {scenario_change:+.2f}%",
                transform=ax.transAxes,
                fontsize=7.5,
                color=COLORS["scenario"],
            )
        elif component in SHOCK_COICOPS:
            delta_yoy = shock_size * float(lag0_map.loc[component])
            scenario_index = baseline_index * np.exp(delta_yoy)
            scenario_change = 100.0 * np.expm1(delta_yoy)
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
                f"AR-only scenario vs baseline: {scenario_change:+.2f}%",
                transform=ax.transAxes,
                fontsize=7.5,
                color=COLORS["scenario"],
            )
        else:
            ax.text(
                0.03,
                0.06,
                "no pass-through multiplier",
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
        title = (
            f"{component} (shock source)"
            if component in SHOCK_COICOPS
            else component
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        if component not in beta_table.index:
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
            label=(
                "Scenario: CP045 + CP0722 "
                f"(+{shock_size:.2f} log pts each)"
            ),
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
        f"{country}: simultaneous CP045 + CP0722 shock across HICP components",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Both CP045 and CP0722 receive a "
        f"+{shock_size:.2f} log-point shock at the origin. Baselines are "
        "conditional on realized shocks for multiplier targets; CP045 and "
        "CP0722 use AR-only baselines, with their own lag-0 AR coefficient "
        "applied for the scenario.",
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
        "--shock-size",
        type=float,
        default=DEFAULT_SHOCK_SIZE,
        help=(
            "Shock in log points applied to both CP045 and CP0722; "
            "0.10 means a 10 percent shock to each."
        ),
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
    baselines, multipliers, history = load_artifacts(
        args.baselines,
        args.multipliers,
        args.history,
    )
    output = Path(
        args.output
        or f"scenario_jointfit_{country}_CP045_CP0722.png"
    )
    path = make_plot(
        baselines,
        multipliers,
        history,
        country,
        args.shock_size,
        output,
    )
    print(path)


if __name__ == "__main__":
    main()

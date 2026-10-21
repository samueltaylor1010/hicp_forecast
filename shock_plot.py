"""Plot a CP045 level shock across all components of one country."""

import argparse
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hicp_var_dynamic import HORIZON, simulate_path

DB_PATH = "hicp.db"
ARTIFACT_PATH = "hicp_var_dynamic.joblib"
DEFAULT_HISTORY_YEARS = 5
DEFAULT_SOURCE = "CP045"
DEFAULT_SHOCK_PCT = 10.0
COLORS = {
    "history": "#1f4e79",
    "baseline": "#e67e22",
    "scenario": "#c0392b",
}


def load_history(db_path, geo):
    """Load the complete historical panel for one geography."""
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT time_period, coicop, hicp_index
            FROM hicp_panel_data
            WHERE geo = ?
            ORDER BY time_period
            """,
            conn,
            params=(geo,),
            parse_dates=["time_period"],
        )
    if frame.empty:
        raise ValueError(f"No history found for {geo}")
    return frame.pivot(index="time_period", columns="coicop", values="hicp_index")


def make_plot(
    history,
    country,
    source,
    shock_pct,
    start,
    output,
):
    """Create the 15-panel historical, baseline, and scenario chart."""
    components = country["components"]
    if source not in components:
        raise ValueError(f"Component {source} is not available for {country['geo']}")

    origin = pd.Timestamp(country["latest_origin"])
    if origin not in history.index:
        raise ValueError(f"Missing historical index at forecast origin {origin.date()}")
    if start >= origin:
        raise ValueError("Start date must be before the forecast origin")
    if start not in history.index:
        start = history.index[history.index >= start].min()
    if pd.isna(start):
        raise ValueError("No historical observations in the requested window")

    recent = history.loc[start:origin, components]
    if recent.isna().any().any():
        raise ValueError("Historical window contains missing component data")

    source_idx = components.index(source)
    shock_log = np.log1p(shock_pct / 100.0)
    state = country["state"].to_numpy(dtype=float)
    transition = country["transition"]
    intercepts = country["intercepts"].to_numpy(dtype=float)
    month_effects = country["month_effects"].to_numpy(dtype=float)
    base_origin = history.loc[origin, components].to_numpy(dtype=float)

    baseline_z = simulate_path(
        transition,
        intercepts,
        month_effects,
        state,
        origin,
        horizon=HORIZON,
    )
    scenario_z = simulate_path(
        transition,
        intercepts,
        month_effects,
        state,
        origin,
        horizon=HORIZON,
        shocked_component=source_idx,
        shock_size=shock_log,
    )

    forecast_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=HORIZON, freq="MS"
    )
    baseline_levels = base_origin * np.exp(np.cumsum(baseline_z, axis=0))
    scenario_origin = base_origin.copy()
    scenario_origin[source_idx] *= np.exp(shock_log)
    scenario_levels = scenario_origin * np.exp(np.cumsum(scenario_z, axis=0))

    plot_components = [source] + [c for c in components if c != source]
    component_index = {component: idx for idx, component in enumerate(components)}

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(20, 10.5),
        sharex=True,
        constrained_layout=False,
    )
    axes = axes.ravel()
    zoom_start = origin - pd.DateOffset(months=9)
    zoom_history = history.loc[zoom_start:origin, components]

    for ax, component in zip(axes, plot_components):
        idx = component_index[component]
        is_source = component == source
        ax.plot(
            recent.index,
            recent[component],
            color=COLORS["history"],
            linewidth=1.5,
        )
        ax.plot(
            [origin] + list(forecast_dates),
            [base_origin[idx]] + baseline_levels[:, idx].tolist(),
            color=COLORS["baseline"],
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=3,
        )
        ax.plot(
            [origin] + list(forecast_dates),
            [scenario_origin[idx]] + scenario_levels[:, idx].tolist(),
            color=COLORS["scenario"],
            linewidth=2.0,
            marker="o",
            markersize=3,
        )
        ax.axvline(origin, color="#7f8c8d", linestyle=":", linewidth=1.0)
        ax.grid(alpha=0.2)
        ax.set_title(
            f"{component} (shocked +{shock_pct:.1f}%)"
            if is_source
            else component,
            fontsize=10,
            fontweight="bold" if is_source else "normal",
        )
        if is_source:
            ax.set_facecolor("#fff5f0")

        cumulative_change = 100.0 * (
            scenario_levels[-1, idx] / baseline_levels[-1, idx] - 1.0
        )
        ax.text(
            0.02,
            0.95,
            f"3m scenario vs baseline: {cumulative_change:+.2f}%",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            color=COLORS["scenario"],
        )

        inset = ax.inset_axes([0.52, 0.08, 0.44, 0.28])
        inset.plot(
            zoom_history.index,
            zoom_history[component],
            color=COLORS["history"],
            linewidth=1.0,
        )
        inset.plot(
            [origin] + list(forecast_dates),
            [base_origin[idx]] + baseline_levels[:, idx].tolist(),
            color=COLORS["baseline"],
            linewidth=1.2,
            linestyle="--",
            marker="o",
            markersize=2,
        )
        inset.plot(
            [origin] + list(forecast_dates),
            [scenario_origin[idx]] + scenario_levels[:, idx].tolist(),
            color=COLORS["scenario"],
            linewidth=1.3,
            marker="o",
            markersize=2,
        )
        inset.axvline(origin, color="#7f8c8d", linestyle=":", linewidth=0.8)
        inset.grid(alpha=0.15)
        inset.tick_params(labelsize=6, length=2)
        inset.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))

    for row in range(3):
        axes[row * 5].set_ylabel("Index (2025 = 100)")
    for ax in axes[-5:]:
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    handles = [
        Line2D([0], [0], color=COLORS["history"], linewidth=2, label="Historical"),
        Line2D(
            [0],
            [0],
            color=COLORS["baseline"],
            linewidth=2,
            linestyle="--",
            marker="o",
            label="Baseline forecast",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["scenario"],
            linewidth=2,
            marker="o",
            label="Shock scenario",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=10,
    )
    fig.suptitle(
        f"{country['geo']}: {source} +{shock_pct:.1f}% level shock — "
        f"historical, baseline, and scenario",
        fontsize=16,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.06,
        hspace=0.35,
        wspace=0.25,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", required=True, help="Geography code, e.g. AT")
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE, help="Shocked component, default CP045"
    )
    parser.add_argument(
        "--shock-pct",
        type=float,
        default=DEFAULT_SHOCK_PCT,
        help="Permanent current-level shock in percent",
    )
    parser.add_argument(
        "--start",
        help="History start date, e.g. 2021-01; defaults to five years",
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to hicp.db")
    parser.add_argument(
        "--artifact", default=ARTIFACT_PATH, help="Path to dynamic VAR artifact"
    )
    parser.add_argument("--output", help="Output PNG path")
    return parser.parse_args()


def main():
    """Generate the historical, baseline, and shock-scenario grid."""
    args = parse_args()
    geo = args.geo.upper()
    source = args.source.upper()
    if args.shock_pct <= -100:
        raise ValueError("Shock percentage must be greater than -100")

    bundle = joblib.load(args.artifact)
    country = bundle["countries"].get(geo)
    if country is None:
        raise ValueError(f"Geography {geo} is not in the artifact")

    history = load_history(args.db, geo)
    origin = pd.Timestamp(country["latest_origin"])
    start = (
        pd.to_datetime(args.start).to_period("M").to_timestamp()
        if args.start
        else origin - pd.DateOffset(years=DEFAULT_HISTORY_YEARS)
    )
    output = args.output or f"shock_plot_{geo}_{source}.png"
    path = make_plot(history, country, source, args.shock_pct, start, output)
    print(path)


if __name__ == "__main__":
    main()

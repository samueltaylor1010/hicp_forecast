"""Matplotlib figures for the HICP shock dashboard."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

try:
    from .labels import country_label
    from .scenario_engine import (
        AggregationResult,
        DashboardData,
        SHOCK_COICOPS,
        component_history,
        component_levels,
        headline_history,
        selected_headline,
    )
except ImportError:  # Shinylive loads the app directory as top-level modules.
    from labels import country_label
    from scenario_engine import (
        AggregationResult,
        DashboardData,
        SHOCK_COICOPS,
        component_history,
        component_levels,
        headline_history,
        selected_headline,
    )

COLORS = {
    "history": "#1f4e79",
    "baseline": "#e67e22",
    "scenario": "#c0392b",
    "origin": "#7f8c8d",
    "shocked": "#8e44ad",
}


def headline_figure(
    data: DashboardData,
    aggregation: AggregationResult,
    geo: str,
) -> Figure:
    history = headline_history(data, geo)
    headline = selected_headline(aggregation, geo)
    target = data.target_period
    origin = data.origin

    fig = Figure(figsize=(9.2, 6.9), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(
        history["date"],
        history["index_value"],
        color=COLORS["history"],
        linewidth=1.9,
    )
    ax.axvline(
        origin,
        color=COLORS["origin"],
        linestyle=":",
        linewidth=1.0,
    )

    last_origin_value = float(history["index_value"].iloc[-1])
    ax.plot(
        [origin, target],
        [last_origin_value, headline["baseline"]],
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )
    ax.plot(
        [origin, target],
        [last_origin_value, headline["scenario"]],
        color=COLORS["scenario"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

    ax.scatter(
        [target],
        [headline["baseline"]],
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=0.7,
        s=70,
        zorder=4,
    )
    ax.scatter(
        [target],
        [headline["scenario"]],
        color=COLORS["scenario"],
        edgecolor="white",
        linewidth=0.7,
        marker="*",
        s=180,
        zorder=5,
    )

    ax.text(
        0.02,
        0.06,
        (
            f"Scenario vs baseline: {headline['delta']:+.2f} index pts "
            f"({headline['pct_change']:+.2f}%)"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["scenario"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )

    ax.set_title(
        f"{country_label(geo)}: headline HICP index",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.set_ylabel("Index")
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=9)
    locator = mdates.MonthLocator(interval=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["history"],
            linewidth=2,
            label="Historical headline index",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["baseline"],
            marker="o",
            linestyle="None",
            markersize=7,
            label=f"Conditional baseline t+3 ({target:%Y-%m})",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["scenario"],
            marker="*",
            linestyle="None",
            markersize=11,
            label="Scenario",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        ncol=1,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.13)
    return fig


def component_grid_figure(
    data: DashboardData,
    aggregation: AggregationResult,
    geo: str,
    shock_percent: float | None,
) -> Figure:
    components = [
        *[code for code in SHOCK_COICOPS if code in data.coicops],
        *[
            code
            for code in sorted(data.coicops)
            if code not in SHOCK_COICOPS
        ],
    ]
    history = component_history(data, geo).reindex(columns=components)
    levels = component_levels(aggregation, geo).reindex(components)
    target = data.target_period
    origin = data.origin

    ncols = 3
    nrows = int(np.ceil(len(components) / ncols))
    fig = Figure(figsize=(10.0, 3.15 * nrows), dpi=100)
    axes = fig.subplots(nrows, ncols, sharex=True, squeeze=False).ravel()

    for ax, component in zip(axes, components):
        series = history[component].dropna()
        ax.plot(
            series.index,
            series.to_numpy(),
            color=COLORS["history"],
            linewidth=1.5,
        )

        baseline = float(levels.loc[component, "baseline"])
        scenario = float(levels.loc[component, "scenario"])
        ax.scatter(
            [target],
            [baseline],
            color=COLORS["baseline"],
            edgecolor="white",
            linewidth=0.6,
            marker="o",
            s=40,
            zorder=4,
        )
        ax.scatter(
            [target],
            [scenario],
            color=COLORS["scenario"],
            edgecolor="white",
            linewidth=0.6,
            marker="*",
            s=95,
            zorder=5,
        )

        if component in SHOCK_COICOPS and geo != "EA":
            origin_value = float(series.loc[origin])
            percent = 0.0 if shock_percent is None else shock_percent
            shocked_value = origin_value * (1.0 + percent / 100.0)
            ax.scatter(
                [origin],
                [origin_value],
                color=COLORS["origin"],
                edgecolor="white",
                linewidth=0.6,
                s=32,
                zorder=4,
            )
            ax.scatter(
                [origin],
                [shocked_value],
                color=COLORS["shocked"],
                edgecolor="white",
                linewidth=0.6,
                s=42,
                zorder=5,
            )

        change = scenario - baseline
        label = "AR-only" if component in SHOCK_COICOPS else "Pass-through"
        ax.text(
            0.03,
            0.06,
            f"{label}: {change:+.2f} pts",
            transform=ax.transAxes,
            fontsize=7.4,
            color=COLORS["scenario"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "pad": 1.5,
            },
        )

        ax.axvline(
            origin,
            color=COLORS["origin"],
            linestyle=":",
            linewidth=0.9,
        )
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=7)
        ax.set_title(
            component,
            fontsize=9,
            fontweight="bold",
            pad=3,
        )

    for ax in axes[len(components):]:
        ax.set_visible(False)

    for ax in axes[(nrows - 1) * ncols:]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for row in range(nrows):
        axes[row * ncols].set_ylabel("Index", fontsize=8)

    source_note = (
        "Purple origin marker is the shocked latest index; gray is the "
        "unshocked origin."
    )
    fig.suptitle(
        (
            f"{country_label(geo)}: component scenarios for a "
            f"{shock_percent:+.0f}% CP045 + CP0722 origin-index shock"
            if shock_percent is not None and geo != "EA"
            else f"{country_label(geo)}: component scenarios"
        ),
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.995,
    )
    if geo != "EA":
        fig.text(0.01, 0.005, source_note, fontsize=7.5, color="#555555")
        fig.subplots_adjust(
            left=0.055,
            right=0.99,
            top=0.955,
            bottom=0.035,
            hspace=0.55,
            wspace=0.24,
        )
    else:
        fig.subplots_adjust(
            left=0.055,
            right=0.99,
            top=0.965,
            bottom=0.035,
            hspace=0.55,
            wspace=0.24,
        )
    return fig

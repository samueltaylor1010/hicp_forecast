"""Dark matplotlib figures for the HICP shock dashboard."""

from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

try:
    from .labels import component_label, country_label
    from .scenario_engine import DashboardData, GraphSeries
except ImportError:  # Shinylive loads the app directory as top-level modules.
    from labels import component_label, country_label
    from scenario_engine import DashboardData, GraphSeries

BG = "#0b1118"
AX_BG = "#0f1822"
TEXT = "#e6edf3"
MUTED = "#93a4b8"
GRID = "#263445"
HISTORY = "#60a5fa"
BASELINE = "#fbbf24"
SCENARIO = "#f87171"
ORIGIN = "#94a3b8"
SHOCKED = "#c084fc"


def _style_axes(ax, fig: Figure) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.7)


def series_figure(
    data: DashboardData,
    series_data: GraphSeries,
    geo: str,
) -> Figure:
    """Render one headline or component series with baseline and scenario."""
    history = series_data.history
    target = data.target_period
    origin = data.origin

    if series_data.code == "HEADLINE":
        title = f"{country_label(geo)}: headline HICP index"
    else:
        title = (
            f"{country_label(geo)}: "
            f"{component_label(series_data.code)} "
            f"({series_data.code})"
        )

    fig = Figure(figsize=(9.2, 6.9), dpi=100)
    ax = fig.add_subplot(111)
    _style_axes(ax, fig)

    ax.plot(
        history["date"],
        history["index_value"],
        color=HISTORY,
        linewidth=1.9,
    )
    ax.axvline(
        origin,
        color=ORIGIN,
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
    )

    origin_value = series_data.origin_index
    ax.plot(
        [origin, target],
        [origin_value, series_data.baseline],
        color=BASELINE,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )
    ax.plot(
        [origin, target],
        [origin_value, series_data.scenario],
        color=SCENARIO,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

    ax.scatter(
        [target],
        [series_data.baseline],
        color=BASELINE,
        edgecolor=BG,
        linewidth=0.7,
        s=70,
        zorder=4,
    )
    ax.scatter(
        [target],
        [series_data.scenario],
        color=SCENARIO,
        edgecolor=BG,
        linewidth=0.7,
        marker="*",
        s=180,
        zorder=5,
    )

    if (
        series_data.is_source
        and series_data.shock_percent is not None
        and geo != "EA"
    ):
        shocked_origin = origin_value * (
            1.0 + series_data.shock_percent / 100.0
        )
        ax.scatter(
            [origin],
            [origin_value],
            color=ORIGIN,
            edgecolor=BG,
            linewidth=0.6,
            s=36,
            zorder=4,
        )
        ax.scatter(
            [origin],
            [shocked_origin],
            color=SHOCKED,
            edgecolor=BG,
            linewidth=0.6,
            s=48,
            zorder=5,
        )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        loc="left",
        color=TEXT,
    )
    ax.set_ylabel("Index", color=TEXT)
    locator = mdates.MonthLocator(interval=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    handles = [
        Line2D(
            [0],
            [0],
            color=HISTORY,
            linewidth=2,
            label="Historical index",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE,
            marker="o",
            linestyle="None",
            markersize=7,
            label=f"Conditional baseline t+3 ({target:%Y-%m})",
        ),
        Line2D(
            [0],
            [0],
            color=SCENARIO,
            marker="*",
            linestyle="None",
            markersize=11,
            label="Scenario",
        ),
    ]
    if (
        series_data.is_source
        and series_data.shock_percent is not None
        and geo != "EA"
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                color=SHOCKED,
                marker="o",
                linestyle="None",
                markersize=6,
                label="Shocked origin index",
            )
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor=AX_BG,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8.5,
        ncol=1,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.13)
    return fig

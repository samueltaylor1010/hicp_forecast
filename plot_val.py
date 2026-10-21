"""Plot historical HICP values and the three-month Huber forecast.

Reads ``hicp_panel_data`` and ``hicp_huber_forecasts`` for one
``(geo, coicop)`` pair and draws the history and forecast as separate
coloured line segments.
"""

import argparse
import sqlite3

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DB_PATH = "hicp.db"
ACTUAL_TABLE = "hicp_panel_data"
FORECAST_TABLE = "hicp_huber_forecasts"

ACTUAL_COLOR = "#1f77b4"
FORECAST_COLOR = "#d62728"
BOUNDARY_COLOR = "0.6"


def load_plot_data(
    geo: str,
    coicop: str,
    db_path: str = DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load actual history and forecasts for one series.

    Raises:
        ValueError: If either the history or the forecast is absent.
    """
    with sqlite3.connect(db_path) as conn:
        actual = pd.read_sql_query(
            f"SELECT time_period, hicp_index FROM {ACTUAL_TABLE} "
            "WHERE geo = ? AND coicop = ? ORDER BY time_period",
            conn,
            params=[geo, coicop],
            parse_dates=["time_period"],
        )
        forecast = pd.read_sql_query(
            f"SELECT time_period, horizon, forecast FROM {FORECAST_TABLE} "
            "WHERE geo = ? AND coicop = ? ORDER BY horizon",
            conn,
            params=[geo, coicop],
            parse_dates=["time_period"],
        )

    if actual.empty:
        raise ValueError(f"No historical data for geo={geo!r}, coicop={coicop!r}.")
    if forecast.empty:
        raise ValueError(f"No forecast for geo={geo!r}, coicop={coicop!r}.")
    return actual, forecast


def plot_series(
    geo: str,
    coicop: str,
    db_path: str = DB_PATH,
    start: str | None = None,
    end: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    dpi: int = 150,
    save_path: str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot actual history and the forecast for one ``(geo, coicop)`` pair.

    Args:
        geo: Geography code, for example ``"EA"``.
        coicop: COICOP component, for example ``"TOTAL"``.
        db_path: SQLite database path.
        start: Optional inclusive start date; defaults to five years of history.
        end: Optional inclusive end date; defaults to the last forecast month.
        ax: Optional existing axes to draw on.
        figsize: Figure size when creating new axes.
        dpi: Figure resolution when creating new axes.
        save_path: Optional output image path.
        show: Whether to call ``matplotlib.pyplot.show()``.

    Returns:
        The figure and axes containing the plot.
    """
    actual, forecast = load_plot_data(geo, coicop, db_path)
    start_ts = (
        pd.Timestamp(start)
        if start
        else actual["time_period"].max() - pd.DateOffset(years=5)
    )
    end_ts = (
        pd.Timestamp(end) if end else forecast["time_period"].max()
    )

    actual = actual[actual["time_period"].between(start_ts, end_ts)]
    forecast = forecast[forecast["time_period"].between(start_ts, end_ts)]
    if actual.empty:
        raise ValueError("No historical data in the requested date range.")
    if forecast.empty:
        raise ValueError("No forecast in the requested date range.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.plot(
        actual["time_period"],
        actual["hicp_index"],
        color=ACTUAL_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Actual",
    )

    bridge = pd.concat(
        [
            actual.tail(1)[["time_period", "hicp_index"]].rename(
                columns={"hicp_index": "value"}
            ),
            forecast[["time_period", "forecast"]].rename(
                columns={"forecast": "value"}
            ),
        ],
        ignore_index=True,
    )
    ax.plot(
        bridge["time_period"],
        bridge["value"],
        color=FORECAST_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="Forecast",
    )

    ax.axvline(
        actual["time_period"].max(),
        color=BOUNDARY_COLOR,
        linestyle=":",
        linewidth=1.0,
        zorder=0,
    )

    ax.set_xlim(start_ts, end_ts)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("HICP index")
    ax.set_title(f"{geo} / {coicop} — actual vs 3-month forecast")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot actual HICP history and the Huber forecast."
    )
    parser.add_argument("geo", help="Geography code, e.g. EA.")
    parser.add_argument("coicop", help="COICOP component, e.g. TOTAL.")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path.")
    parser.add_argument("--start", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--save", help="Output image path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line plotting utility."""
    args = parse_args()
    save_path = args.save or f"plot_val_{args.geo}_{args.coicop}.png"
    plot_series(
        args.geo,
        args.coicop,
        db_path=args.db,
        start=args.start,
        end=args.end,
        save_path=save_path,
        show=args.show,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()

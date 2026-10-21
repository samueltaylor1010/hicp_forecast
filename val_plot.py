"""Developer plotting helpers for fitted HICP STL + SARIMA series.

This module is a local debugging tool and is intentionally not part of the
final project. It reads the fitted tables produced by ``hicp_ts_fit.py`` and
plots the raw observations, the robust STL baseline, and the SARIMA fit and
forecast for any ``(geo, coicop)`` pair.
"""

import argparse
import sqlite3
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DB_PATH = "hicp.db"
OBS_TABLE = "hicp_ts_observations"
FORECAST_TABLE = "hicp_ts_forecasts"
MODEL_TABLE = "hicp_ts_models"
PLOT_START = "2019-01-01"
PLOT_END = "2026-12-01"

ACTUAL_COLOR = "#1f77b4"
OUTLIER_COLOR = "#d62728"
STL_COLOR = "#2ca02c"
SARIMA_COLOR = "#ff7f0e"
CI_COLOR = "#ff7f0e"


@dataclass(frozen=True)
class SeriesPlotData:
    """Fitted observations, forecasts, and metadata for one series."""

    geo: str
    coicop: str
    observations: pd.DataFrame
    forecasts: pd.DataFrame
    model: pd.Series | None


def list_series(
    db_path: str = DB_PATH,
    status: str | None = None,
) -> pd.DataFrame:
    """Return the fitted series available in the database.

    Args:
        db_path: SQLite database path.
        status: Optional status filter, for example ``"ok"``.

    Returns:
        Frame with one row per fitted ``(geo, coicop)`` pair.
    """
    query = (
        "SELECT geo, coicop, status, arima_order, seasonal_order, "
        f"train_start, train_end FROM {MODEL_TABLE}"
    )
    params: list[str] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY geo, coicop"

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def load_plot_data(
    geo: str,
    coicop: str,
    db_path: str = DB_PATH,
) -> SeriesPlotData:
    """Load the fitted observations, forecasts, and model row for a series.

    Args:
        geo: Geography code, for example ``"EA"``.
        coicop: COICOP code, for example ``"TOTAL"``.
        db_path: SQLite database path.

    Returns:
        Parsed plot data for the requested pair.

    Raises:
        ValueError: If the pair has no fitted observations or forecasts.
    """
    with sqlite3.connect(db_path) as conn:
        observations = pd.read_sql_query(
            f"SELECT * FROM {OBS_TABLE} WHERE geo = ? AND coicop = ? "
            "ORDER BY time_period",
            conn,
            params=[geo, coicop],
        )
        forecasts = pd.read_sql_query(
            f"SELECT * FROM {FORECAST_TABLE} WHERE geo = ? AND coicop = ? "
            "ORDER BY time_period",
            conn,
            params=[geo, coicop],
        )
        model_rows = pd.read_sql_query(
            f"SELECT * FROM {MODEL_TABLE} WHERE geo = ? AND coicop = ?",
            conn,
            params=[geo, coicop],
        )

    if observations.empty and forecasts.empty:
        raise ValueError(
            f"No fitted data for geo={geo!r}, coicop={coicop!r}. "
            "Use list_series() to see available pairs."
        )

    observations = observations.assign(
        time_period=pd.to_datetime(observations["time_period"])
    )
    forecasts = forecasts.assign(
        time_period=pd.to_datetime(forecasts["time_period"])
    )
    model = None if model_rows.empty else model_rows.iloc[0]

    return SeriesPlotData(
        geo=geo,
        coicop=coicop,
        observations=observations,
        forecasts=forecasts,
        model=model,
    )


def plot_series(
    geo: str,
    coicop: str,
    db_path: str = DB_PATH,
    ax: Axes | None = None,
    start: str = PLOT_START,
    end: str = PLOT_END,
    figsize: tuple[float, float] = (14.0, 6.0),
    dpi: int = 150,
    save_path: str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot observations, STL baseline, and SARIMA fit/forecast for a series.

    Actual observations are scatter points split into normal and outlier
    colours. The robust STL baseline is drawn as a solid line, the in-sample
    SARIMA fit joins the forecast, and the forecast carries a 95% band.

    Args:
        geo: Geography code, for example ``"EA"``.
        coicop: COICOP code, for example ``"TOTAL"``.
        db_path: SQLite database path.
        ax: Optional existing axes to draw on.
        start: Inclusive x-axis start date.
        end: Inclusive x-axis end date.
        figsize: Figure size in inches when creating new axes.
        dpi: Figure resolution when creating new axes.
        save_path: Optional output image path.
        show: Whether to call ``matplotlib.pyplot.show()``.

    Returns:
        The figure and axes containing the plot.
    """
    data = load_plot_data(geo, coicop, db_path)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    observations = data.observations[
        data.observations["time_period"].between(start_ts, end_ts)
    ]
    forecasts = data.forecasts[
        data.forecasts["time_period"].between(start_ts, end_ts)
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    outliers = observations["is_outlier"].astype(bool)
    normal = observations[~outliers]
    flagged = observations[outliers]

    ax.scatter(
        normal["time_period"],
        normal["actual"],
        s=18,
        color=ACTUAL_COLOR,
        label="Actual",
        zorder=4,
    )
    if not flagged.empty:
        ax.scatter(
            flagged["time_period"],
            flagged["actual"],
            s=30,
            color=OUTLIER_COLOR,
            label="Outlier",
            zorder=5,
        )

    stl = observations.dropna(subset=["stl_baseline"])
    if not stl.empty:
        ax.plot(
            stl["time_period"],
            stl["stl_baseline"],
            color=STL_COLOR,
            linewidth=1.8,
            label="STL baseline",
            zorder=3,
        )

    fitted = observations.dropna(subset=["sarima_fitted"])
    if not fitted.empty:
        ax.plot(
            fitted["time_period"],
            fitted["sarima_fitted"],
            color=SARIMA_COLOR,
            linewidth=1.5,
            label="SARIMA fitted",
            zorder=2,
        )

    if not forecasts.empty:
        forecast_values = forecasts["forecast"]
        ax.plot(
            forecasts["time_period"],
            forecast_values,
            color=SARIMA_COLOR,
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=4,
            label="SARIMA forecast",
            zorder=4,
        )
        if forecasts[["lower_95", "upper_95"]].notna().all(axis=None):
            ax.fill_between(
                forecasts["time_period"],
                forecasts["lower_95"],
                forecasts["upper_95"],
                color=CI_COLOR,
                alpha=0.2,
                label="95% CI",
                zorder=1,
            )
        else:
            ax.axvline(
                forecasts["time_period"].min(),
                color="0.6",
                linestyle=":",
                linewidth=0.8,
                zorder=0,
            )

    ax.set_xlim(start_ts, end_ts)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("HICP index")
    ax.grid(alpha=0.3)

    title = f"{geo} / {coicop} — STL + SARIMA"
    if data.model is not None:
        order = data.model.get("arima_order")
        seasonal = data.model.get("seasonal_order")
        status = data.model.get("status")
        if pd.notna(order):
            title += f"\nARIMA{order} x {seasonal}"
        if status != "ok":
            title += f"  [{status}]"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot fitted STL + SARIMA results for one HICP series."
    )
    parser.add_argument("geo", nargs="?", help="Geography code, e.g. EA.")
    parser.add_argument("coicop", nargs="?", help="COICOP code, e.g. TOTAL.")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--start", default=PLOT_START, help="Inclusive x-axis start date."
    )
    parser.add_argument(
        "--end", default=PLOT_END, help="Inclusive x-axis end date."
    )
    parser.add_argument("--save", help="Output image path.")
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_series",
        help="List available fitted series and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line plotting helper."""
    args = parse_args()

    if args.list_series:
        print(list_series(args.db).to_string(index=False))
        return

    if not args.geo or not args.coicop:
        raise SystemExit("Provide both geo and coicop, or use --list.")

    save_path = args.save or f"val_plot_{args.geo}_{args.coicop}.png"

    plot_series(
        args.geo,
        args.coicop,
        db_path=args.db,
        start=args.start,
        end=args.end,
        save_path=save_path,
        show=False,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()

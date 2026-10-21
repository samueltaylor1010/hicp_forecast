"""Plot historical HICP data and the three-month component forecast."""

import argparse
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB_PATH = "hicp.db"
ARTIFACT_PATH = "hicp_var_models.joblib"
DEFAULT_HISTORY_YEARS = 5


def load_history(db_path, geo, coicop):
    """Load the historical index for one geography and component."""
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT time_period, hicp_index
            FROM hicp_panel_data
            WHERE geo = ? AND coicop = ?
            ORDER BY time_period
            """,
            conn,
            params=(geo, coicop),
            parse_dates=["time_period"],
        )
    if frame.empty:
        raise ValueError(f"No history found for {geo} {coicop}")
    return frame.set_index("time_period")["hicp_index"]


def load_forecast(artifact_path, geo, coicop):
    """Load and validate the forecast path for one component."""
    bundle = joblib.load(artifact_path)
    country = bundle["countries"].get(geo)
    if country is None:
        raise ValueError(f"Geography {geo} is not in the model artifact")
    if coicop not in country["components"]:
        raise ValueError(f"Component {coicop} is not available for {geo}")

    rows = []
    for horizon in sorted(country["horizons"]):
        result = country["horizons"][horizon]
        if coicop not in result["forecast_log"].index:
            raise ValueError(f"Missing h={horizon} forecast for {coicop}")
        rows.append(
            {
                "horizon": horizon,
                "period": pd.Timestamp(result["forecast_period"]),
                "log_change": float(result["forecast_log"].loc[coicop]),
            }
        )

    forecast = pd.DataFrame(rows).sort_values("period")
    if len(forecast) != 3:
        raise ValueError(f"Expected three forecast horizons for {geo} {coicop}")
    if not np.isfinite(forecast["log_change"]).all():
        raise ValueError(f"Non-finite forecast for {geo} {coicop}")

    origin = pd.Timestamp(country["horizons"][1]["forecast_origin"])
    return origin, forecast


def make_plot(history, origin, forecast, geo, coicop, start, output):
    """Create and save the historical-plus-forecast index plot."""
    history = history.loc[:origin]
    if origin not in history.index:
        raise ValueError(f"Missing index value at forecast origin {origin.date()}")
    base_value = float(history.loc[origin])
    forecast = forecast.copy()
    forecast["index"] = base_value * np.exp(forecast["log_change"])
    forecast["pct"] = 100.0 * np.expm1(forecast["log_change"])

    if start >= origin:
        raise ValueError("Start date must be before the forecast origin")
    recent = history.loc[start:]
    if recent.empty:
        raise ValueError("No historical observations in the requested window")

    plot_dates = [origin] + forecast["period"].tolist()
    plot_values = [base_value] + forecast["index"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        recent.index,
        recent.to_numpy(),
        color="#1f4e79",
        linewidth=1.8,
        label="Historical HICP index",
    )
    ax.plot(
        plot_dates,
        plot_values,
        color="#c0392b",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=6,
        label="Forecast path",
    )
    ax.axvline(origin, color="#7f8c8d", linestyle=":", linewidth=1.0)

    summary = "\n".join(
        f"{row['period']:%Y-%m}: {row['pct']:+.2f}%"
        for _, row in forecast.iterrows()
    )
    ax.text(
        0.99,
        0.03,
        "Cumulative forecast\n" + summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#c0392b",
            "alpha": 0.9,
        },
    )

    ax.set_title(
        f"{geo} {coicop}: historical HICP index and three-month forecast",
        fontsize=13,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Index (2025 = 100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", required=True, help="Geography code, e.g. AT")
    parser.add_argument("--coicop", required=True, help="COICOP code, e.g. CP01")
    parser.add_argument("--db", default=DB_PATH, help="Path to hicp.db")
    parser.add_argument(
        "--artifact", default=ARTIFACT_PATH, help="Path to model artifact"
    )
    parser.add_argument(
        "--start",
        help="History start date, e.g. 2021-01; defaults to five years at month start",
    )
    parser.add_argument("--output", help="Output PNG path")
    return parser.parse_args()


def main():
    """Generate the requested validation chart."""
    args = parse_args()
    geo = args.geo.upper()
    coicop = args.coicop.upper()
    history = load_history(args.db, geo, coicop)
    origin, forecast = load_forecast(args.artifact, geo, coicop)

    if args.start:
        start = pd.to_datetime(args.start)
        start = start.to_period("M").to_timestamp()
    else:
        start = origin - pd.DateOffset(years=DEFAULT_HISTORY_YEARS)

    output = args.output or f"val_plot_{geo}_{coicop}.png"
    path = make_plot(history, origin, forecast, geo, coicop, start, output)
    print(path)


if __name__ == "__main__":
    main()

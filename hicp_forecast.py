"""Forecast HICP components and estimate energy shock pass-through.

The pipeline keeps the 21 euro-area member countries and the mutually
exclusive COICOP components.  It fits a direct three-month-ahead AR model
for every country-component series, estimates constrained-free local
projection multipliers for CP045 and CP0722 shocks, and writes three
human-auditable CSV files.
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

DB_PATH = "hicp.db"
HORIZON = 3
MIN_OBS = 60
SHOCKS = ("CP045", "CP0722")
EXPECTED_GEOS = (
    "AT",
    "BE",
    "BG",
    "CY",
    "DE",
    "EE",
    "EL",
    "ES",
    "FI",
    "FR",
    "HR",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PT",
    "SI",
    "SK",
)
RETAINED_COICOPS = (
    "CP01",
    "CP02",
    "CP03",
    "CP045",
    "CP04_CORE",
    "CP05",
    "CP06",
    "CP0722",
    "CP07_CORE",
    "CP08",
    "CP09",
    "CP10",
    "CP11",
    "CP12",
    "CP13",
)
TARGET_COICOPS = tuple(
    coicop for coicop in RETAINED_COICOPS if coicop not in SHOCKS
)
BASELINE_FILE = "baselines.csv"
MULTIPLIER_FILE = "multipliers.csv"
HISTORY_FILE = "historical_indices.csv"
HISTORY_MONTHS = 36


def load_panel(db_path: str | Path) -> pd.DataFrame:
    """Load and validate the eligible HICP panel from SQLite."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    with sqlite3.connect(path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT time_period, geo, coicop, hicp_index
            FROM hicp_panel_data
            WHERE geo != 'EA'
              AND coicop NOT IN ('TOTAL', 'CP04', 'CP07')
            ORDER BY geo, coicop, time_period
            """,
            conn,
            parse_dates=["time_period"],
        )

    if frame.empty:
        raise ValueError("No eligible HICP observations found")

    geos = set(frame["geo"])
    if geos != set(EXPECTED_GEOS):
        raise ValueError(
            f"Unexpected geographies: {sorted(geos.symmetric_difference(EXPECTED_GEOS))}"
        )

    coicops = set(frame["coicop"])
    if coicops != set(RETAINED_COICOPS):
        raise ValueError(
            f"Unexpected COICOP codes: {sorted(coicops.symmetric_difference(RETAINED_COICOPS))}"
        )

    keys = ["time_period", "geo", "coicop"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate country-component-date observations found")
    if frame[["hicp_index"]].isna().any().any():
        raise ValueError("Missing HICP index values found")
    if (frame["hicp_index"] <= 0).any():
        raise ValueError("HICP indices must be strictly positive")

    for (geo, coicop), group in frame.groupby(["geo", "coicop"], sort=False):
        dates = pd.DatetimeIndex(group["time_period"].sort_values())
        expected = pd.date_range(dates.min(), dates.max(), freq="MS")
        if not dates.equals(expected):
            raise ValueError(f"Interior monthly gap in {geo}/{coicop}")

    return frame.sort_values(keys).reset_index(drop=True)


def build_panel(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Pivot the panel, compute YoY log differences, and find the origin."""
    wide = frame.pivot(
        index="time_period",
        columns=["geo", "coicop"],
        values="hicp_index",
    ).sort_index()
    columns = pd.MultiIndex.from_product(
        [EXPECTED_GEOS, RETAINED_COICOPS],
        names=["geo", "coicop"],
    )
    wide = wide.reindex(columns=columns)

    if wide.isna().all().any():
        missing = wide.columns[wide.isna().all()].tolist()
        raise ValueError(f"Series with no observations: {missing}")
    if (wide.dropna() <= 0).any().any():
        raise ValueError("HICP indices must be strictly positive")

    yoy = np.log(wide / wide.shift(12))
    complete = wide.notna().all(axis=1)
    if not complete.any():
        raise ValueError("No month has observations for every retained series")
    origin = wide.index[complete].max()

    required_dates = [
        origin - pd.DateOffset(months=offset) for offset in range(4)
    ]
    if not all(date in yoy.index for date in required_dates):
        raise ValueError("Insufficient history at the forecast origin")
    if yoy.loc[required_dates].isna().any().any():
        raise ValueError("Missing YoY history at the forecast origin")

    return wide, yoy, origin


def training_frame(series: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """Build direct-horizon target and four contemporaneous/lagged regressors."""
    return pd.DataFrame(
        {
            "target": series.shift(-horizon),
            "lag0": series,
            "lag1": series.shift(1),
            "lag2": series.shift(2),
            "lag3": series.shift(3),
        }
    ).dropna()


def fit_baseline(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    geo: str,
    coicop: str,
    origin: pd.Timestamp,
) -> dict:
    """Fit one direct AR model and invert its YoY forecast to an index."""
    series = yoy[(geo, coicop)]
    data = training_frame(series)
    if len(data) < MIN_OBS:
        raise ValueError(
            f"Insufficient observations for {geo}/{coicop}: {len(data)}"
        )

    design = np.column_stack(
        [
            np.ones(len(data)),
            data[["lag0", "lag1", "lag2", "lag3"]].to_numpy(dtype=float),
        ]
    )
    target = data["target"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    forecast_row = np.array(
        [
            1.0,
            *[
                float(series.loc[origin - pd.DateOffset(months=offset)])
                for offset in range(4)
            ],
        ]
    )
    yoy_forecast = float(forecast_row @ coefficients)
    if not np.isfinite(yoy_forecast):
        raise ValueError(f"Non-finite forecast for {geo}/{coicop}")

    target_period = origin + pd.DateOffset(months=HORIZON)
    base_period = target_period - pd.DateOffset(months=12)
    base_index = float(wide.loc[base_period, (geo, coicop)])
    baseline_index = base_index * np.exp(yoy_forecast)

    if not np.isfinite(baseline_index) or baseline_index <= 0:
        raise ValueError(f"Invalid baseline index for {geo}/{coicop}")

    return {
        "country": geo,
        "coicop": coicop,
        "forecast_origin": origin,
        "target_period": target_period,
        "baseline_index_t3": baseline_index,
    }


def fit_multiplier(
    yoy: pd.DataFrame,
    geo: str,
    target_coicop: str,
    shock_variable: str,
) -> dict:
    """Fit one unconstrained local projection with all terms estimated jointly."""
    target_series = yoy[(geo, target_coicop)]
    shock_series = yoy[(geo, shock_variable)]
    data = pd.DataFrame(
        {
            "target": target_series.shift(-HORIZON),
            "shock": shock_series,
            "lag0": target_series,
            "lag1": target_series.shift(1),
            "lag2": target_series.shift(2),
            "lag3": target_series.shift(3),
        }
    ).dropna()

    if len(data) < MIN_OBS:
        raise ValueError(
            f"Insufficient observations for {geo}/{target_coicop} "
            f"on {shock_variable}: {len(data)}"
        )

    design = np.column_stack(
        [
            np.ones(len(data)),
            data["shock"].to_numpy(dtype=float),
            data[["lag0", "lag1", "lag2", "lag3"]].to_numpy(dtype=float),
        ]
    )
    target = data["target"].to_numpy(dtype=float)
    bounds = (
        np.full(design.shape[1], -np.inf),
        np.full(design.shape[1], np.inf),
    )
    result = lsq_linear(design, target, bounds=bounds, max_iter=1000)
    if not result.success:
        raise RuntimeError(
            f"Local projection failed for {geo}/{target_coicop} "
            f"on {shock_variable}: {result.message}"
        )

    beta = float(result.x[1])
    if not np.isfinite(beta):
        raise ValueError(
            f"Non-finite multiplier for {geo}/{target_coicop} "
            f"on {shock_variable}"
        )

    return {
        "country": geo,
        "target_coicop": target_coicop,
        "shock_variable": shock_variable,
        "beta_multiplier": beta,
    }


def build_baselines(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Build the baseline forecast table for every retained series."""
    rows = [
        fit_baseline(wide, yoy, geo, coicop, origin)
        for geo in EXPECTED_GEOS
        for coicop in RETAINED_COICOPS
    ]
    return pd.DataFrame(rows).sort_values(["country", "coicop"])


def build_multipliers(yoy: pd.DataFrame) -> pd.DataFrame:
    """Build pass-through multipliers for both shocks and all target components."""
    rows = [
        fit_multiplier(yoy, geo, target, shock)
        for geo in EXPECTED_GEOS
        for shock in SHOCKS
        for target in TARGET_COICOPS
    ]
    return pd.DataFrame(rows).sort_values(
        ["country", "target_coicop", "shock_variable"]
    )


def build_history(wide: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Return the last 36 monthly index observations for all retained series."""
    start = origin - pd.DateOffset(months=HISTORY_MONTHS - 1)
    recent = wide.loc[start:origin].copy()
    recent.index.name = "date"
    recent.columns = [
        f"{geo}|{coicop}" for geo, coicop in recent.columns
    ]
    history = (
        recent.reset_index()
        .melt(id_vars="date", var_name="series", value_name="index_value")
        .assign(
            country=lambda frame: frame["series"].str.split("|").str[0],
            coicop=lambda frame: frame["series"].str.split("|").str[1],
        )
        .drop(columns="series")
    )
    return history[
        ["country", "coicop", "date", "index_value"]
    ].sort_values(["country", "coicop", "date"])


def validate_artifacts(
    baselines: pd.DataFrame,
    multipliers: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    """Check shapes, keys, and values before serialization."""
    n_countries = len(EXPECTED_GEOS)
    n_series = len(RETAINED_COICOPS)
    n_targets = len(TARGET_COICOPS)

    expected_baselines = n_countries * n_series
    expected_multipliers = n_countries * len(SHOCKS) * n_targets
    expected_history = n_countries * n_series * HISTORY_MONTHS

    if len(baselines) != expected_baselines:
        raise ValueError(
            f"Expected {expected_baselines} baselines, got {len(baselines)}"
        )
    if len(multipliers) != expected_multipliers:
        raise ValueError(
            f"Expected {expected_multipliers} multipliers, got {len(multipliers)}"
        )
    if len(history) != expected_history:
        raise ValueError(
            f"Expected {expected_history} historical rows, got {len(history)}"
        )

    baseline_keys = ["country", "coicop"]
    multiplier_keys = ["country", "target_coicop", "shock_variable"]
    history_keys = ["country", "coicop", "date"]

    if baselines.duplicated(baseline_keys).any():
        raise ValueError("Duplicate baseline keys")
    if multipliers.duplicated(multiplier_keys).any():
        raise ValueError("Duplicate multiplier keys")
    if history.duplicated(history_keys).any():
        raise ValueError("Duplicate historical keys")

    if set(baselines["coicop"]) != set(RETAINED_COICOPS):
        raise ValueError("Unexpected components in baseline output")
    if set(multipliers["shock_variable"]) != set(SHOCKS):
        raise ValueError("Unexpected shocks in multiplier output")
    if set(multipliers["target_coicop"]) != set(TARGET_COICOPS):
        raise ValueError("Unexpected targets in multiplier output")
    if baselines["country"].nunique() != n_countries:
        raise ValueError("Unexpected countries in baseline output")
    if "EA" in set(history["country"]):
        raise ValueError("Aggregate geography EA found in historical output")
    if set(history["coicop"]) & {"TOTAL", "CP04", "CP07"}:
        raise ValueError("Aggregate components found in historical output")

    if not np.isfinite(baselines["baseline_index_t3"]).all():
        raise ValueError("Non-finite baseline indices")
    if not (baselines["baseline_index_t3"] > 0).all():
        raise ValueError("Non-positive baseline indices")
    if not np.isfinite(multipliers["beta_multiplier"]).all():
        raise ValueError("Non-finite multipliers")
    if not (history["index_value"] > 0).all():
        raise ValueError("Non-positive historical indices")


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
) -> None:
    """Serialize a table with stable column order and date formatting."""
    frame[columns].to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10g",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--out-dir", default=".", help="Directory for CSV artifacts."
    )
    return parser.parse_args()


def main() -> None:
    """Run the forecasting and pass-through pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    panel = load_panel(args.db)
    wide, yoy, origin = build_panel(panel)
    logging.info(
        "Loaded %s countries, %s components, origin %s",
        len(EXPECTED_GEOS),
        len(RETAINED_COICOPS),
        origin.date(),
    )

    baselines = build_baselines(wide, yoy, origin)
    multipliers = build_multipliers(yoy)
    history = build_history(wide, origin)
    validate_artifacts(baselines, multipliers, history)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        baselines,
        out_dir / BASELINE_FILE,
        ["country", "coicop", "forecast_origin", "target_period", "baseline_index_t3"],
    )
    write_csv(
        multipliers,
        out_dir / MULTIPLIER_FILE,
        ["country", "target_coicop", "shock_variable", "beta_multiplier"],
    )
    write_csv(
        history,
        out_dir / HISTORY_FILE,
        ["country", "coicop", "date", "index_value"],
    )

    logging.info("Wrote %s baseline rows", len(baselines))
    logging.info("Wrote %s multiplier rows", len(multipliers))
    logging.info("Wrote %s historical rows", len(history))
    logging.info("Artifacts written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()

"""Forecast HICP components and estimate energy shock pass-through.

The pipeline keeps the 21 euro-area member countries and the mutually
exclusive COICOP components.  Each non-energy target is fitted once on
both CP045 and CP0722 YoY shocks plus its own lags.  That single joint fit
produces the conditional baseline and both pass-through multipliers.
CP045 and CP0722 remain excluded as multiplier targets and receive
AR-only baselines.
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


def ar_training_frame(series: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """Build a direct-horizon target and four own-lag regressors."""
    return pd.DataFrame(
        {
            "target": series.shift(-horizon),
            "lag0": series,
            "lag1": series.shift(1),
            "lag2": series.shift(2),
            "lag3": series.shift(3),
        }
    ).dropna()


def invert_yoy_forecast(
    wide: pd.DataFrame,
    geo: str,
    coicop: str,
    origin: pd.Timestamp,
    yoy_forecast: float,
) -> float:
    """Convert a target-period YoY log forecast into an index level."""
    target_period = origin + pd.DateOffset(months=HORIZON)
    base_period = target_period - pd.DateOffset(months=12)
    base_index = float(wide.loc[base_period, (geo, coicop)])
    baseline_index = base_index * np.exp(yoy_forecast)
    if not np.isfinite(baseline_index) or baseline_index <= 0:
        raise ValueError(f"Invalid baseline index for {geo}/{coicop}")
    return baseline_index


def fit_ar_baseline(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    geo: str,
    coicop: str,
    origin: pd.Timestamp,
) -> dict:
    """Fit an AR-only baseline for an excluded energy target."""
    series = yoy[(geo, coicop)]
    data = ar_training_frame(series)
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

    return {
        "country": geo,
        "coicop": coicop,
        "forecast_origin": origin,
        "target_period": origin + pd.DateOffset(months=HORIZON),
        "baseline_index_t3": invert_yoy_forecast(
            wide, geo, coicop, origin, yoy_forecast
        ),
    }


def fit_joint_model(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    geo: str,
    target_coicop: str,
    origin: pd.Timestamp,
) -> tuple[dict, list[dict]]:
    """Fit one joint shock model and derive its baseline and both multipliers."""
    target_series = yoy[(geo, target_coicop)]
    shock_045 = yoy[(geo, "CP045")]
    shock_0722 = yoy[(geo, "CP0722")]
    data = pd.DataFrame(
        {
            "target": target_series.shift(-HORIZON),
            "shock_045": shock_045,
            "shock_0722": shock_0722,
            "lag0": target_series,
            "lag1": target_series.shift(1),
            "lag2": target_series.shift(2),
            "lag3": target_series.shift(3),
        }
    ).dropna()

    if len(data) < MIN_OBS:
        raise ValueError(
            f"Insufficient observations for {geo}/{target_coicop}: {len(data)}"
        )

    design = np.column_stack(
        [
            np.ones(len(data)),
            data[["shock_045", "shock_0722"]].to_numpy(dtype=float),
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
            f"Joint model failed for {geo}/{target_coicop}: {result.message}"
        )

    coefficients = result.x
    forecast_row = np.array(
        [
            1.0,
            float(shock_045.loc[origin]),
            float(shock_0722.loc[origin]),
            *[
                float(target_series.loc[origin - pd.DateOffset(months=offset)])
                for offset in range(4)
            ],
        ]
    )
    yoy_forecast = float(forecast_row @ coefficients)
    if not np.isfinite(yoy_forecast):
        raise ValueError(f"Non-finite forecast for {geo}/{target_coicop}")

    baseline = {
        "country": geo,
        "coicop": target_coicop,
        "forecast_origin": origin,
        "target_period": origin + pd.DateOffset(months=HORIZON),
        "baseline_index_t3": invert_yoy_forecast(
            wide, geo, target_coicop, origin, yoy_forecast
        ),
    }
    multipliers = [
        {
            "country": geo,
            "target_coicop": target_coicop,
            "shock_variable": shock,
            "beta_multiplier": float(coefficients[index]),
        }
        for index, shock in ((1, "CP045"), (2, "CP0722"))
    ]

    if not all(np.isfinite(row["beta_multiplier"]) for row in multipliers):
        raise ValueError(f"Non-finite multiplier for {geo}/{target_coicop}")

    return baseline, multipliers


def build_artifacts(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each series once and build baseline and multiplier tables."""
    baseline_rows = []
    multiplier_rows = []

    for geo in EXPECTED_GEOS:
        for target_coicop in TARGET_COICOPS:
            baseline, multipliers = fit_joint_model(
                wide, yoy, geo, target_coicop, origin
            )
            baseline_rows.append(baseline)
            multiplier_rows.extend(multipliers)

        for coicop in SHOCKS:
            baseline_rows.append(
                fit_ar_baseline(wide, yoy, geo, coicop, origin)
            )

    baselines = pd.DataFrame(baseline_rows).sort_values(["country", "coicop"])
    multipliers = pd.DataFrame(multiplier_rows).sort_values(
        ["country", "target_coicop", "shock_variable"]
    )
    return baselines, multipliers


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
    """Run the joint forecasting and pass-through pipeline."""
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

    baselines, multipliers = build_artifacts(wide, yoy, origin)
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

    logging.info(
        "Fitted %s joint target models and %s AR energy baselines",
        len(EXPECTED_GEOS) * len(TARGET_COICOPS),
        len(EXPECTED_GEOS) * len(SHOCKS),
    )
    logging.info("Wrote %s baseline rows", len(baselines))
    logging.info("Wrote %s multiplier rows", len(multipliers))
    logging.info("Wrote %s historical rows", len(history))
    logging.info("Artifacts written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()

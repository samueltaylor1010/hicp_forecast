"""Forecast HICP components and estimate energy shock pass-through.

The pipeline keeps the 21 euro-area member countries and the mutually
exclusive COICOP components. Each non-energy target is fitted once on both
CP045 and CP0722 YoY shocks plus its own lags. That single joint fit produces
the conditional baseline and both pass-through multipliers. CP045 and CP0722
remain excluded as multiplier targets and receive AR-only baselines.
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
BASELINE_COLUMNS = [
    "country",
    "coicop",
    "forecast_origin",
    "target_period",
    "baseline_index_t3",
]
MULTIPLIER_COLUMNS = [
    "country",
    "target_coicop",
    "shock_variable",
    "beta_multiplier",
]
HISTORY_COLUMNS = ["country", "coicop", "date", "index_value"]


def _validate_panel(frame: pd.DataFrame) -> None:
    """Validate the scope and monthly integrity of the loaded panel.

    Args:
        frame (pd.DataFrame): Long panel with ``time_period``, ``geo``,
            ``coicop``, and ``hicp_index`` columns.

    Raises:
        ValueError: If the panel scope, index values, or monthly continuity
            is invalid.
    """
    geos = set(frame["geo"])
    if geos != set(EXPECTED_GEOS):
        unexpected = sorted(geos.symmetric_difference(EXPECTED_GEOS))
        raise ValueError(f"Unexpected geographies: {unexpected}")

    coicops = set(frame["coicop"])
    if coicops != set(RETAINED_COICOPS):
        unexpected = sorted(coicops.symmetric_difference(RETAINED_COICOPS))
        raise ValueError(f"Unexpected COICOP codes: {unexpected}")

    if frame["hicp_index"].isna().any():
        raise ValueError("Missing HICP index values found")
    if (frame["hicp_index"] <= 0).any():
        raise ValueError("HICP indices must be strictly positive")

    for (geo, coicop), group in frame.groupby(["geo", "coicop"], sort=False):
        dates = pd.DatetimeIndex(group["time_period"].sort_values())
        expected = pd.date_range(dates.min(), dates.max(), freq="MS")
        if not dates.equals(expected):
            raise ValueError(f"Interior monthly gap in {geo}/{coicop}")


def load_panel(db_path: str | Path) -> pd.DataFrame:
    """Load the eligible HICP panel from SQLite.

    Args:
        db_path (str | Path): Path to the SQLite database.

    Returns:
        pd.DataFrame: Sorted long panel with eligible countries and components.

    Raises:
        FileNotFoundError: If the database file does not exist.
        ValueError: If no eligible observations are found or validation fails.
    """
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

    _validate_panel(frame)
    return frame.sort_values(
        ["time_period", "geo", "coicop"]
    ).reset_index(drop=True)


def _find_origin(wide: pd.DataFrame) -> pd.Timestamp:
    """Find the latest month observed for every retained series.

    Args:
        wide (pd.DataFrame): Wide index panel indexed by month.

    Returns:
        pd.Timestamp: Latest month with no missing values across all series.

    Raises:
        ValueError: If no month is complete across all series.
    """
    complete = wide.notna().all(axis=1)
    if not complete.any():
        raise ValueError("No month has observations for every retained series")
    return wide.index[complete].max()


def build_panel(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Pivot the panel, compute YoY log differences, and find the origin.

    Args:
        frame (pd.DataFrame): Validated long HICP panel.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]: Wide index panel,
        YoY log differences, and the common forecast origin.

    Raises:
        ValueError: If any country-component series is empty or origin
            history is incomplete.
    """
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

    yoy = np.log(wide / wide.shift(12))
    origin = _find_origin(wide)

    required_dates = [
        origin - pd.DateOffset(months=offset) for offset in range(4)
    ]
    if not all(date in yoy.index for date in required_dates):
        raise ValueError("Insufficient history at the forecast origin")
    if yoy.loc[required_dates].isna().any().any():
        raise ValueError("Missing YoY history at the forecast origin")

    return wide, yoy, origin


def _training_frame(
    series: pd.Series,
    shocks: dict[str, pd.Series] | None = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Build a direct-horizon target, optional shocks, and four own lags.

    Args:
        series (pd.Series): Target YoY series.
        shocks (dict[str, pd.Series] | None): Optional shock series keyed by
            COICOP code.
        horizon (int): Direct forecast horizon in months.

    Returns:
        pd.DataFrame: Complete training rows containing ``target``, optional
        ``shock_*`` columns, and ``lag0`` through ``lag3``.
    """
    data = {
        "target": series.shift(-horizon),
        "lag0": series,
        "lag1": series.shift(1),
        "lag2": series.shift(2),
        "lag3": series.shift(3),
    }
    for coicop, shock in (shocks or {}).items():
        data[f"shock_{coicop}"] = shock
    return pd.DataFrame(data).dropna()


def _forecast_features(
    series: pd.Series,
    shocks: dict[str, pd.Series],
    origin: pd.Timestamp,
) -> np.ndarray:
    """Build the origin-valued predictor row for a direct forecast.

    Args:
        series (pd.Series): Target YoY series.
        shocks (dict[str, pd.Series]): Shock series keyed by COICOP code.
        origin (pd.Timestamp): Forecast origin.

    Returns:
        np.ndarray: Predictor row ordered as intercept, shocks, then own lags.
    """
    return np.array(
        [
            1.0,
            *[float(shock.loc[origin]) for shock in shocks.values()],
            *[
                float(series.loc[origin - pd.DateOffset(months=offset)])
                for offset in range(4)
            ],
        ]
    )


def invert_yoy_forecast(
    wide: pd.DataFrame,
    geo: str,
    coicop: str,
    origin: pd.Timestamp,
    yoy_forecast: float,
) -> float:
    """Convert a target-period YoY log forecast into an index level.

    Args:
        wide (pd.DataFrame): Wide index panel.
        geo (str): Country code.
        coicop (str): Component code.
        origin (pd.Timestamp): Forecast origin.
        yoy_forecast (float): Forecasted YoY log difference at ``origin + 3``.

    Returns:
        float: Forecast index level at ``origin + 3``.
    """
    target_period = origin + pd.DateOffset(months=HORIZON)
    base_period = target_period - pd.DateOffset(months=12)
    base_index = float(wide.loc[base_period, (geo, coicop)])
    return base_index * np.exp(yoy_forecast)


def _solve_model(
    data: pd.DataFrame,
    shock_names: tuple[str, ...],
) -> np.ndarray:
    """Estimate one direct model, using the appropriate OLS solver branch.

    Args:
        data (pd.DataFrame): Training rows with target, lags, and optional
            shock columns.
        shock_names (tuple[str, ...]): Shock COICOP codes included in the
            model. An empty tuple fits the AR-only baseline.

    Returns:
        np.ndarray: Estimated coefficient vector.

    Raises:
        RuntimeError: If the bounded least-squares solver fails.
    """
    predictor_columns = [
        *[f"shock_{shock}" for shock in shock_names],
        "lag0",
        "lag1",
        "lag2",
        "lag3",
    ]
    design = np.column_stack(
        [
            np.ones(len(data)),
            data[predictor_columns].to_numpy(dtype=float),
        ]
    )
    target = data["target"].to_numpy(dtype=float)

    if not shock_names:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return coefficients

    result = lsq_linear(
        design,
        target,
        bounds=(
            np.full(design.shape[1], -np.inf),
            np.full(design.shape[1], np.inf),
        ),
        max_iter=1000,
    )
    if not result.success:
        raise RuntimeError(f"Joint model failed: {result.message}")
    return result.x


def fit_direct_model(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    geo: str,
    coicop: str,
    origin: pd.Timestamp,
    shock_coicops: tuple[str, ...] = (),
) -> tuple[dict, list[dict]]:
    """Fit one direct model and derive its baseline and multipliers.

    Args:
        wide (pd.DataFrame): Wide index panel.
        yoy (pd.DataFrame): Wide YoY log-difference panel.
        geo (str): Country code.
        coicop (str): Target component code.
        origin (pd.Timestamp): Forecast origin.
        shock_coicops (tuple[str, ...]): Shock components to include. An
            empty tuple fits the AR-only energy baseline.

    Returns:
        tuple[dict, list[dict]]: Baseline row and multiplier rows.

    Raises:
        ValueError: If too few complete training observations are available.
        RuntimeError: If the shock model solver fails.
    """
    series = yoy[(geo, coicop)]
    shocks = {shock: yoy[(geo, shock)] for shock in shock_coicops}
    data = _training_frame(series, shocks)
    if len(data) < MIN_OBS:
        raise ValueError(
            f"Insufficient observations for {geo}/{coicop}: {len(data)}"
        )

    coefficients = _solve_model(data, tuple(shock_coicops))
    forecast_row = _forecast_features(series, shocks, origin)
    yoy_forecast = float(forecast_row @ coefficients)

    baseline = {
        "country": geo,
        "coicop": coicop,
        "forecast_origin": origin,
        "target_period": origin + pd.DateOffset(months=HORIZON),
        "baseline_index_t3": invert_yoy_forecast(
            wide, geo, coicop, origin, yoy_forecast
        ),
    }
    multipliers = [
        {
            "country": geo,
            "target_coicop": coicop,
            "shock_variable": shock,
            "beta_multiplier": float(coefficients[index]),
        }
        for index, shock in enumerate(shock_coicops, start=1)
    ]
    return baseline, multipliers


def build_artifacts(
    wide: pd.DataFrame,
    yoy: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every retained series once and build output tables.

    Args:
        wide (pd.DataFrame): Wide index panel.
        yoy (pd.DataFrame): Wide YoY log-difference panel.
        origin (pd.Timestamp): Forecast origin.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Baseline and multiplier tables.
    """
    baseline_rows = []
    multiplier_rows = []

    for geo in EXPECTED_GEOS:
        for coicop in RETAINED_COICOPS:
            shocks = SHOCKS if coicop in TARGET_COICOPS else ()
            baseline, multipliers = fit_direct_model(
                wide, yoy, geo, coicop, origin, shocks
            )
            baseline_rows.append(baseline)
            multiplier_rows.extend(multipliers)

    baselines = pd.DataFrame(baseline_rows).sort_values(["country", "coicop"])
    multipliers = pd.DataFrame(multiplier_rows).sort_values(
        ["country", "target_coicop", "shock_variable"]
    )
    return baselines, multipliers


def build_history(wide: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Return the last 36 monthly index observations for all series.

    Args:
        wide (pd.DataFrame): Wide index panel.
        origin (pd.Timestamp): Forecast origin and end of the history window.

    Returns:
        pd.DataFrame: Long historical index table.
    """
    start = origin - pd.DateOffset(months=HISTORY_MONTHS - 1)
    recent = wide.loc[start:origin].rename_axis(
        index="date",
        columns=["country", "coicop"],
    )
    history = (
        recent.stack(["country", "coicop"], future_stack=True)
        .rename("index_value")
        .reset_index()
    )
    return history[HISTORY_COLUMNS].sort_values(
        ["country", "coicop", "date"]
    )


def validate_artifacts(
    baselines: pd.DataFrame,
    multipliers: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    """Check output shapes, keys, and value domains before serialization.

    Args:
        baselines (pd.DataFrame): Baseline forecast table.
        multipliers (pd.DataFrame): Multiplier table.
        history (pd.DataFrame): Historical index table.

    Raises:
        ValueError: If row counts, key uniqueness, or value domains are invalid.
    """
    expected_rows = {
        "baselines": len(EXPECTED_GEOS) * len(RETAINED_COICOPS),
        "multipliers": (
            len(EXPECTED_GEOS) * len(SHOCKS) * len(TARGET_COICOPS)
        ),
        "history": (
            len(EXPECTED_GEOS) * len(RETAINED_COICOPS) * HISTORY_MONTHS
        ),
    }
    tables = {
        "baselines": baselines,
        "multipliers": multipliers,
        "history": history,
    }
    for name, frame in tables.items():
        if len(frame) != expected_rows[name]:
            raise ValueError(
                f"Expected {expected_rows[name]} {name} rows, got {len(frame)}"
            )

    if baselines.duplicated(["country", "coicop"]).any():
        raise ValueError("Duplicate baseline keys")
    if multipliers.duplicated(
        ["country", "target_coicop", "shock_variable"]
    ).any():
        raise ValueError("Duplicate multiplier keys")
    if history.duplicated(["country", "coicop", "date"]).any():
        raise ValueError("Duplicate historical keys")

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
    """Serialize a table with stable column order and date formatting.

    Args:
        frame (pd.DataFrame): Table to write.
        path (Path): Destination CSV path.
        columns (list[str]): Output column order.
    """
    frame[columns].to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10g",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed database and output-directory arguments.
    """
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
        BASELINE_COLUMNS,
    )
    write_csv(
        multipliers,
        out_dir / MULTIPLIER_FILE,
        MULTIPLIER_COLUMNS,
    )
    write_csv(
        history,
        out_dir / HISTORY_FILE,
        HISTORY_COLUMNS,
    )

    logging.info("Wrote %s baseline rows", len(baselines))
    logging.info("Wrote %s multiplier rows", len(multipliers))
    logging.info("Wrote %s historical rows", len(history))
    logging.info("Artifacts written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()

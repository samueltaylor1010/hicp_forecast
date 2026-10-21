"""Fit robust STL + SARIMA models to each HICP series.

Each (geo, coicop) series is decomposed with a robust STL model. Only
isolated outlier runs at or below --max-outlier-run are replaced by the STL
baseline; consecutive runs and the final --protect-last observations are
retained. An auto-ARIMA model is then fitted to the scrubbed series. Results
are written to three tables in the HICP database: fitted observations,
three-month forecasts, and model metadata.
"""

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import STL

DB_PATH = "hicp.db"
OBS_TABLE = "hicp_ts_observations"
FORECAST_TABLE = "hicp_ts_forecasts"
MODEL_TABLE = "hicp_ts_models"

OBS_DDL = """
CREATE TABLE {table} (
    time_period DATE NOT NULL,
    geo VARCHAR(10) NOT NULL,
    coicop VARCHAR(20) NOT NULL,
    actual FLOAT,
    stl_trend FLOAT,
    stl_seasonal FLOAT,
    stl_resid FLOAT,
    stl_baseline FLOAT,
    is_outlier INTEGER NOT NULL,
    is_scrubbed INTEGER NOT NULL,
    scrubbed FLOAT,
    sarima_fitted FLOAT,
    PRIMARY KEY (time_period, geo, coicop)
)
"""

FORECAST_DDL = """
CREATE TABLE {table} (
    time_period DATE NOT NULL,
    geo VARCHAR(10) NOT NULL,
    coicop VARCHAR(20) NOT NULL,
    horizon INTEGER NOT NULL,
    forecast FLOAT NOT NULL,
    lower_95 FLOAT,
    upper_95 FLOAT,
    PRIMARY KEY (time_period, geo, coicop)
)
"""

MODEL_DDL = """
CREATE TABLE {table} (
    geo VARCHAR(10) NOT NULL,
    coicop VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    error TEXT,
    n_obs INTEGER,
    n_gaps INTEGER,
    n_outliers INTEGER,
    stl_period INTEGER,
    stl_robust INTEGER,
    robust_sigma FLOAT,
    outlier_threshold FLOAT,
    arima_order VARCHAR(20),
    seasonal_order VARCHAR(30),
    aic FLOAT,
    aicc FLOAT,
    bic FLOAT,
    n_train INTEGER,
    train_start DATE,
    train_end DATE,
    forecast_start DATE,
    forecast_end DATE,
    forecast_horizon INTEGER,
    params_json TEXT,
    elapsed_seconds FLOAT,
    fitted_at TEXT NOT NULL,
    PRIMARY KEY (geo, coicop)
)
"""

MODEL_COLUMNS = [
    "geo",
    "coicop",
    "status",
    "error",
    "n_obs",
    "n_gaps",
    "n_outliers",
    "stl_period",
    "stl_robust",
    "robust_sigma",
    "outlier_threshold",
    "arima_order",
    "seasonal_order",
    "aic",
    "aicc",
    "bic",
    "n_train",
    "train_start",
    "train_end",
    "forecast_start",
    "forecast_end",
    "forecast_horizon",
    "params_json",
    "elapsed_seconds",
    "fitted_at",
]


@dataclass(frozen=True)
class Config:
    period: int = 12
    horizon: int = 3
    min_obs: int = 36
    max_outlier_run: int = 1
    protect_last: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit STL + SARIMA models to HICP series."
    )
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--geos",
        help="Comma-separated geography codes (default: all).",
    )
    parser.add_argument(
        "--coicops",
        help="Comma-separated COICOP codes (default: all).",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=12,
        help="Seasonal period for STL and SARIMA (default: 12).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecast horizon in months (default: 3).",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=36,
        help="Minimum observations required per series (default: 36).",
    )
    parser.add_argument(
        "--max-outlier-run",
        type=int,
        default=1,
        help="Maximum consecutive outlier months to scrub (default: 1).",
    )
    parser.add_argument(
        "--protect-last",
        type=int,
        default=1,
        help="Number of most recent observations never to scrub (default: 1).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes (default: -1).",
    )
    return parser.parse_args()


def split_codes(value: str | None) -> list[str] | None:
    """Turn a comma-separated option into a list of codes."""
    if not value:
        return None
    return [code.strip() for code in value.split(",") if code.strip()]


def load_series(
    conn: sqlite3.Connection,
    geos: list[str] | None,
    coicops: list[str] | None,
    min_obs: int,
) -> dict[tuple[str, str], tuple[pd.Series, int]]:
    """Load complete monthly HICP series and gap counts by (geo, coicop)."""
    query = (
        "SELECT time_period, geo, coicop, hicp_index "
        "FROM hicp_panel_data"
    )
    clauses = []
    params = []
    if geos:
        clauses.append(f"geo IN ({','.join('?' for _ in geos)})")
        params.extend(geos)
    if coicops:
        clauses.append(f"coicop IN ({','.join('?' for _ in coicops)})")
        params.extend(coicops)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY geo, coicop, time_period"

    frame = pd.read_sql_query(query, conn, params=params)
    series_by_key = {}

    for (geo, coicop), group in frame.groupby(["geo", "coicop"], sort=True):
        values = group.set_index(pd.to_datetime(group["time_period"]))["hicp_index"]
        values = values.astype(float).sort_index()
        values = values[~values.index.duplicated(keep="first")].dropna()
        if values.empty:
            continue

        full_index = pd.date_range(values.index.min(), values.index.max(), freq="MS")
        n_gaps = len(full_index) - len(values)
        values = values.reindex(full_index)
        if n_gaps:
            values = values.interpolate(method="time", limit_area="inside")
        values = values.dropna().asfreq("MS")

        if len(values) < min_obs:
            logging.warning(
                "Skipping %s/%s: only %d observations", geo, coicop, len(values)
            )
            continue

        values.name = "hicp_index"
        series_by_key[(geo, coicop)] = (values, n_gaps)

    return series_by_key


def empty_observations() -> pd.DataFrame:
    """Return an empty observations frame with the output columns."""
    return pd.DataFrame(
        columns=[
            "time_period",
            "geo",
            "coicop",
            "actual",
            "stl_trend",
            "stl_seasonal",
            "stl_resid",
            "stl_baseline",
            "is_outlier",
            "is_scrubbed",
            "scrubbed",
            "sarima_fitted",
        ]
    )


def empty_forecasts() -> pd.DataFrame:
    """Return an empty forecasts frame with the output columns."""
    return pd.DataFrame(
        columns=[
            "time_period",
            "geo",
            "coicop",
            "horizon",
            "forecast",
            "lower_95",
            "upper_95",
        ]
    )


def fit_series(
    key: tuple[str, str],
    loaded: tuple[pd.Series, int],
    config: Config,
) -> dict:
    """Fit STL and SARIMA for one (geo, coicop) series."""
    geo, coicop = key
    series, n_gaps = loaded
    started = time.perf_counter()
    fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    result = {
        "geo": geo,
        "coicop": coicop,
        "status": "stl_failed",
        "error": None,
        "n_obs": len(series),
        "n_gaps": n_gaps,
        "n_outliers": None,
        "stl_period": config.period,
        "stl_robust": 1,
        "robust_sigma": None,
        "outlier_threshold": None,
        "arima_order": None,
        "seasonal_order": None,
        "aic": None,
        "aicc": None,
        "bic": None,
        "n_train": None,
        "train_start": None,
        "train_end": None,
        "forecast_start": None,
        "forecast_end": None,
        "forecast_horizon": None,
        "params_json": None,
        "elapsed_seconds": None,
        "fitted_at": fitted_at,
        "observations": empty_observations(),
        "forecasts": empty_forecasts(),
    }

    try:
        stl = STL(series, period=config.period, robust=True).fit()
        trend = pd.Series(np.asarray(stl.trend), index=series.index)
        seasonal = pd.Series(np.asarray(stl.seasonal), index=series.index)
        resid = pd.Series(np.asarray(stl.resid), index=series.index)
        baseline = trend + seasonal

        mad = np.median(np.abs(resid - np.median(resid)))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            threshold = 0.0
            outliers = pd.Series(False, index=series.index)
        else:
            threshold = 3.0 * robust_sigma
            outliers = resid.abs() > threshold

        run_id = (outliers != outliers.shift(fill_value=False)).cumsum()
        run_length = outliers.groupby(run_id).transform("size").where(outliers, 0)
        scrub_mask = outliers & (run_length <= config.max_outlier_run)
        if config.protect_last > 0:
            scrub_mask.iloc[-config.protect_last:] = False

        scrubbed = series.copy()
        scrubbed[scrub_mask] = baseline[scrub_mask]

        observations = pd.DataFrame(
            {
                "time_period": series.index.strftime("%Y-%m-%d"),
                "geo": geo,
                "coicop": coicop,
                "actual": series.to_numpy(dtype=float),
                "stl_trend": trend.to_numpy(dtype=float),
                "stl_seasonal": seasonal.to_numpy(dtype=float),
                "stl_resid": resid.to_numpy(dtype=float),
                "stl_baseline": baseline.to_numpy(dtype=float),
                "is_outlier": outliers.astype(int).to_numpy(),
                "is_scrubbed": scrub_mask.astype(int).to_numpy(),
                "scrubbed": scrubbed.to_numpy(dtype=float),
                "sarima_fitted": np.nan,
            }
        )

        result["observations"] = observations
        result["n_outliers"] = int(outliers.sum())
        result["robust_sigma"] = float(robust_sigma)
        result["outlier_threshold"] = float(threshold)
        result["n_train"] = len(series)
        result["train_start"] = series.index[0].strftime("%Y-%m-%d")
        result["train_end"] = series.index[-1].strftime("%Y-%m-%d")
        result["forecast_horizon"] = config.horizon
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

    try:
        model = auto_arima(
            scrubbed.astype(float),
            seasonal=True,
            m=config.period,
            stepwise=True,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            information_criterion="aicc",
            suppress_warnings=True,
            error_action="ignore",
        )

        in_sample = np.asarray(model.predict_in_sample(), dtype=float)
        if len(in_sample) != len(series):
            raise ValueError(
                f"in-sample length {len(in_sample)} != series length {len(series)}"
            )
        observations["sarima_fitted"] = in_sample

        forecast, intervals = model.predict(
            n_periods=config.horizon,
            return_conf_int=True,
            alpha=0.05,
        )
        future_index = pd.date_range(
            start=series.index[-1], periods=config.horizon + 1, freq="MS"
        )[1:]
        forecasts = pd.DataFrame(
            {
                "time_period": future_index.strftime("%Y-%m-%d"),
                "geo": geo,
                "coicop": coicop,
                "horizon": np.arange(1, config.horizon + 1),
                "forecast": np.asarray(forecast, dtype=float),
                "lower_95": np.asarray(intervals[:, 0], dtype=float),
                "upper_95": np.asarray(intervals[:, 1], dtype=float),
            }
        )

        try:
            params = model.arima_res_.params
            if hasattr(params, "to_dict"):
                params = params.to_dict()
            params_json = json.dumps(
                {str(name): float(value) for name, value in params.items()}
            )
        except Exception:
            params_json = None

        result.update(
            {
                "status": "ok",
                "arima_order": str(tuple(model.order)),
                "seasonal_order": str(tuple(model.seasonal_order)),
                "aic": float(model.aic()),
                "aicc": float(model.aicc()),
                "bic": float(model.bic()),
                "forecast_start": future_index[0].strftime("%Y-%m-%d"),
                "forecast_end": future_index[-1].strftime("%Y-%m-%d"),
                "params_json": params_json,
                "observations": observations,
                "forecasts": forecasts,
            }
        )
    except Exception as exc:
        result["status"] = "sarima_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["observations"] = observations

    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def fit_all(
    series_by_key: dict[tuple[str, str], tuple[pd.Series, int]],
    config: Config,
    jobs: int,
) -> list[dict]:
    """Fit every series in parallel and log progress."""
    total = len(series_by_key)
    results = []
    parallel = Parallel(
        n_jobs=jobs,
        prefer="processes",
        return_as="generator_unordered",
    )
    fitted = parallel(
        delayed(fit_series)(key, loaded, config)
        for key, loaded in series_by_key.items()
    )
    for count, result in enumerate(fitted, start=1):
        results.append(result)
        if count % 10 == 0 or count == total:
            logging.info("Fitted %d/%d series", count, total)
    return results


def drop_and_create(conn: sqlite3.Connection, table: str, ddl: str) -> None:
    """Recreate a table from a DDL template."""
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(ddl.format(table=table))
    conn.commit()


def write_results(results: list[dict], db_path: str) -> None:
    """Write results to staging tables, then swap them into place."""
    observations = [
        result["observations"]
        for result in results
        if not result["observations"].empty
    ]
    forecasts = [
        result["forecasts"] for result in results if not result["forecasts"].empty
    ]
    models = pd.DataFrame(
        [
            {column: result[column] for column in MODEL_COLUMNS}
            for result in results
        ],
        columns=MODEL_COLUMNS,
    )

    obs_staging = f"{OBS_TABLE}_staging"
    forecast_staging = f"{FORECAST_TABLE}_staging"
    model_staging = f"{MODEL_TABLE}_staging"

    with sqlite3.connect(db_path) as conn:
        drop_and_create(conn, obs_staging, OBS_DDL)
        drop_and_create(conn, forecast_staging, FORECAST_DDL)
        drop_and_create(conn, model_staging, MODEL_DDL)

        if observations:
            pd.concat(observations, ignore_index=True).to_sql(
                obs_staging, conn, if_exists="append", index=False, chunksize=10_000
            )
        if forecasts:
            pd.concat(forecasts, ignore_index=True).to_sql(
                forecast_staging, conn, if_exists="append", index=False
            )
        models.to_sql(
            model_staging, conn, if_exists="append", index=False
        )
        conn.commit()

        conn.execute(f"DROP TABLE IF EXISTS {OBS_TABLE}")
        conn.execute(f"DROP TABLE IF EXISTS {FORECAST_TABLE}")
        conn.execute(f"DROP TABLE IF EXISTS {MODEL_TABLE}")
        conn.execute(f"ALTER TABLE {obs_staging} RENAME TO {OBS_TABLE}")
        conn.execute(
            f"ALTER TABLE {forecast_staging} RENAME TO {FORECAST_TABLE}"
        )
        conn.execute(f"ALTER TABLE {model_staging} RENAME TO {MODEL_TABLE}")
        conn.execute(
            f"CREATE INDEX idx_{OBS_TABLE}_series "
            f"ON {OBS_TABLE} (geo, coicop, time_period)"
        )
        conn.execute(
            f"CREATE INDEX idx_{FORECAST_TABLE}_series "
            f"ON {FORECAST_TABLE} (geo, coicop, time_period)"
        )
        conn.commit()


def main() -> None:
    """Run the full fitting pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.max_outlier_run < 0:
        raise SystemExit("--max-outlier-run must be non-negative")
    if args.protect_last < 0:
        raise SystemExit("--protect-last must be non-negative")

    config = Config(
        period=args.period,
        horizon=args.horizon,
        min_obs=args.min_obs,
        max_outlier_run=args.max_outlier_run,
        protect_last=args.protect_last,
    )
    geos = split_codes(args.geos)
    coicops = split_codes(args.coicops)

    with sqlite3.connect(args.db) as conn:
        series_by_key = load_series(conn, geos, coicops, config.min_obs)

    if not series_by_key:
        logging.error("No series found for the requested filters.")
        return

    logging.info("Fitting %d series", len(series_by_key))
    results = fit_all(series_by_key, config, args.jobs)
    write_results(results, args.db)

    counts = pd.Series([result["status"] for result in results]).value_counts()
    logging.info("Status counts: %s", counts.to_dict())
    logging.info("Results written to %s", args.db)


if __name__ == "__main__":
    main()

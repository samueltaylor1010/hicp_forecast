"""Build the static dashboard data bundle.

The Shiny app must not read SQLite at runtime, so this script copies the
forecast artefacts and extracts the weights and official headline histories
needed by the dashboard into ``app/data``.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BASELINE_FILE = DATA_DIR / "baselines.csv"
MULTIPLIER_FILE = DATA_DIR / "multipliers.csv"
HISTORY_FILE = DATA_DIR / "historical_indices.csv"
DB_FILE = DATA_DIR / "hicp.db"

COUNTRY_CODES = (
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
COICOP_CODES = (
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
HISTORY_MONTHS = 36


def _load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baselines = pd.read_csv(
        BASELINE_FILE, parse_dates=["forecast_origin", "target_period"]
    )
    multipliers = pd.read_csv(MULTIPLIER_FILE)
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    return baselines, multipliers, history


def _validate_artifacts(
    baselines: pd.DataFrame,
    multipliers: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if baselines.empty:
        raise ValueError("Baselines are empty")
    if set(baselines["country"]) != set(COUNTRY_CODES):
        raise ValueError("Baselines do not contain the expected 21 countries")
    if set(baselines["coicop"]) != set(COICOP_CODES):
        raise ValueError("Baselines do not contain the expected components")
    if baselines["target_period"].nunique() != 1:
        raise ValueError("Baselines contain more than one target period")
    if baselines["forecast_origin"].nunique() != 1:
        raise ValueError("Baselines contain more than one forecast origin")

    target_period = pd.Timestamp(baselines["target_period"].iloc[0])
    origin = pd.Timestamp(baselines["forecast_origin"].iloc[0])
    expected_origin = target_period - pd.DateOffset(months=3)
    if origin != expected_origin:
        raise ValueError(
            f"Forecast origin {origin.date()} is not target - 3 months"
        )

    expected_multiplier_rows = (
        len(COUNTRY_CODES) * len(COICOP_CODES) * 2
        - len(COUNTRY_CODES) * 2 * 2
    )
    if len(multipliers) != expected_multiplier_rows:
        raise ValueError(
            f"Expected {expected_multiplier_rows} multiplier rows, "
            f"got {len(multipliers)}"
        )
    if not np.isfinite(multipliers["beta_multiplier"]).all():
        raise ValueError("Multipliers contain non-finite values")

    expected_history_rows = (
        len(COUNTRY_CODES) * len(COICOP_CODES) * HISTORY_MONTHS
    )
    if len(history) != expected_history_rows:
        raise ValueError(
            f"Expected {expected_history_rows} history rows, got {len(history)}"
        )
    if set(history["country"]) != set(COUNTRY_CODES):
        raise ValueError("History does not contain the expected countries")
    if set(history["coicop"]) != set(COICOP_CODES):
        raise ValueError("History does not contain the expected components")
    if history["date"].max() != origin:
        raise ValueError("History does not end at the forecast origin")
    if history["date"].min() != origin - pd.DateOffset(
        months=HISTORY_MONTHS - 1
    ):
        raise ValueError("History does not cover the expected 36 months")

    return origin, target_period


def _extract_weights(
    db_path: Path,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        available = pd.read_sql_query(
            """
            SELECT MAX(time_period) AS latest
            FROM hicp_panel_data
            WHERE geo != 'EA'
              AND time_period <= ?
            """,
            conn,
            params=[origin.strftime("%Y-%m-%d")],
        )
        latest = available["latest"].iloc[0]
        if latest is None:
            raise ValueError("No weights available at or before the origin")

        placeholders_country = ",".join("?" for _ in COUNTRY_CODES)
        placeholders_coicop = ",".join("?" for _ in COICOP_CODES)
        weights = pd.read_sql_query(
            f"""
            SELECT geo AS country, coicop, item_weight, country_weight
            FROM hicp_panel_data
            WHERE time_period = ?
              AND geo IN ({placeholders_country})
              AND coicop IN ({placeholders_coicop})
            """,
            conn,
            params=[latest, *COUNTRY_CODES, *COICOP_CODES],
        )

    weight_year = int(str(latest)[:4])
    weights["weight_year"] = weight_year
    weights = weights.sort_values(["country", "coicop"]).reset_index(drop=True)

    expected_rows = len(COUNTRY_CODES) * len(COICOP_CODES)
    if len(weights) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} weight rows, got {len(weights)}"
        )
    if weights.duplicated(["country", "coicop"]).any():
        raise ValueError("Duplicate weight keys")
    if not np.isfinite(weights[["item_weight", "country_weight"]]).all().all():
        raise ValueError("Weights contain non-finite values")
    if (weights[["item_weight", "country_weight"]] <= 0).any().any():
        raise ValueError("Weights must be strictly positive")

    item_sums = weights.groupby("country")["item_weight"].sum()
    if not np.allclose(item_sums, 1000.0, atol=0.1):
        raise ValueError("Item weights do not sum to approximately 1000")
    country_sums = weights.groupby("country")["country_weight"].first().sum()
    if not np.isclose(country_sums, 1000.0, atol=0.1):
        raise ValueError("Country weights do not sum to approximately 1000")

    return weights


def _read_history_window(
    db_path: Path,
    origin: pd.Timestamp,
    *,
    geo: str | None,
    coicops: tuple[str, ...],
) -> pd.DataFrame:
    start = origin - pd.DateOffset(months=HISTORY_MONTHS - 1)
    placeholders = ",".join("?" for _ in coicops)
    params: list[object] = [
        start.strftime("%Y-%m-%d"),
        origin.strftime("%Y-%m-%d"),
        *coicops,
    ]
    geo_clause = ""
    if geo is not None:
        geo_clause = "AND geo = ?"
        params.append(geo)

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT geo AS country, coicop, time_period AS date, hicp_index
            FROM hicp_panel_data
            WHERE time_period BETWEEN ? AND ?
              AND coicop IN ({placeholders})
              {geo_clause}
            ORDER BY geo, coicop, time_period
            """,
            conn,
            params=params,
            parse_dates=["date"],
        )


def _extract_headline_history(
    db_path: Path,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    history = _read_history_window(
        db_path,
        origin,
        geo=None,
        coicops=("TOTAL",),
    )
    history = history.rename(columns={"hicp_index": "index_value"})
    history = history[["country", "date", "index_value"]]
    expected_rows = (len(COUNTRY_CODES) + 1) * HISTORY_MONTHS
    if len(history) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} headline history rows, "
            f"got {len(history)}"
        )
    if history.duplicated(["country", "date"]).any():
        raise ValueError("Duplicate headline history keys")
    if not np.isfinite(history["index_value"]).all():
        raise ValueError("Headline history contains non-finite values")
    return history.sort_values(["country", "date"]).reset_index(drop=True)


def _extract_ea_component_history(
    db_path: Path,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    history = _read_history_window(
        db_path,
        origin,
        geo="EA",
        coicops=COICOP_CODES,
    )
    history = history.rename(columns={"hicp_index": "index_value"})
    history = history[["coicop", "date", "index_value"]]
    expected_rows = len(COICOP_CODES) * HISTORY_MONTHS
    if len(history) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} EA component history rows, "
            f"got {len(history)}"
        )
    if history.duplicated(["coicop", "date"]).any():
        raise ValueError("Duplicate EA component history keys")
    if not np.isfinite(history["index_value"]).all():
        raise ValueError("EA component history contains non-finite values")
    return history.sort_values(["coicop", "date"]).reset_index(drop=True)


def _write(frame: pd.DataFrame, path: Path, *, float_format: str = "%.10g") -> None:
    frame.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format=float_format,
    )


def build_app_data(out_dir: Path) -> None:
    baselines, multipliers, history = _load_artifacts()
    origin, target_period = _validate_artifacts(
        baselines, multipliers, history
    )

    weights = _extract_weights(DB_FILE, origin)
    headline_history = _extract_headline_history(DB_FILE, origin)
    ea_component_history = _extract_ea_component_history(DB_FILE, origin)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASELINE_FILE, out_dir / "baselines.csv")
    shutil.copy2(MULTIPLIER_FILE, out_dir / "multipliers.csv")
    shutil.copy2(HISTORY_FILE, out_dir / "historical_indices.csv")
    _write(weights, out_dir / "weights.csv", float_format="%.8f")
    _write(headline_history, out_dir / "headline_history.csv")
    _write(
        ea_component_history,
        out_dir / "ea_component_history.csv",
    )

    print(
        "Dashboard data written to "
        f"{out_dir.resolve()} "
        f"(origin {origin.date()}, target {target_period.date()}, "
        f"weights {weights['weight_year'].iloc[0]})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "app" / "data",
        help="Destination directory for the dashboard CSV bundle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_app_data(parse_args().out_dir)

"""Estimate Jordà local projections for HICP energy shock transmission.

For each geography, shock origin, receiving component, and horizon, fit

    dlog(y_i,t+h) = alpha + beta * dlog(x_s,t) + rho * dlog(y_i,t) + error

with heteroskedasticity-robust (HC0) standard errors. Results are written
to a tidy CSV and a horizon-by-horizon beta matrix.
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

DB_PATH = "hicp.db"
ORIGINS = ("CP045", "CP0722")
EXCLUDED_RECEIVERS = {"TOTAL", "CP04", "CP07"}
HORIZONS = (1, 2, 3)
MIN_OBS = 24
LONG_FILE = "hicp_local_projections.csv"
BETA_FILE = "hicp_local_projections_beta.csv"


def load_growth(db_path: str) -> dict[str, pd.DataFrame]:
    """Load monthly log-differences for every geography and component."""
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            "SELECT time_period, geo, coicop, hicp_index "
            "FROM hicp_panel_data ORDER BY geo, coicop, time_period",
            conn,
            parse_dates=["time_period"],
        )

    growth_by_geo = {}
    for geo, group in frame.groupby("geo", sort=True):
        panel = group.pivot(
            index="time_period", columns="coicop", values="hicp_index"
        ).sort_index()
        if (panel <= 0).any().any():
            raise ValueError(f"Non-positive HICP index for {geo}")

        full_index = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
        panel = panel.reindex(full_index).interpolate(
            method="time", limit_area="inside"
        )
        growth_by_geo[geo] = np.log(panel).diff()

    return growth_by_geo


def build_tasks(
    growth_by_geo: dict[str, pd.DataFrame],
) -> list[tuple[str, str, str, int]]:
    """Build the independent (geo, origin, receiver, horizon) task grid."""
    tasks = []
    for geo, growth in growth_by_geo.items():
        for origin in ORIGINS:
            if origin not in growth.columns:
                continue
            for receiver in sorted(growth.columns):
                if receiver in EXCLUDED_RECEIVERS or receiver == origin:
                    continue
                for horizon in HORIZONS:
                    tasks.append((geo, origin, receiver, horizon))
    return tasks


def fit_task(
    geo: str,
    origin: str,
    receiver: str,
    horizon: int,
    growth: pd.DataFrame,
) -> dict | None:
    """Fit one local projection and return its coefficients and inference."""
    data = pd.DataFrame(
        {
            "target": growth[receiver].shift(-horizon),
            "shock": growth[origin],
            "own_lag": growth[receiver],
        }
    ).dropna()
    if len(data) < MIN_OBS:
        return None

    design = sm.add_constant(data[["shock", "own_lag"]])
    fit = sm.OLS(data["target"], design).fit(cov_type="HC0")

    return {
        "geo": geo,
        "origin_component": origin,
        "receiving_component": receiver,
        "horizon": horizon,
        "alpha": fit.params["const"],
        "beta": fit.params["shock"],
        "rho": fit.params["own_lag"],
        "beta_std_error": fit.bse["shock"],
        "beta_t_stat": fit.tvalues["shock"],
        "beta_p_value": fit.pvalues["shock"],
        "n_obs": int(fit.nobs),
        "r_squared": fit.rsquared,
    }


def fit_all(tasks, growth_by_geo, jobs: int) -> list[dict]:
    """Fit all local projections in parallel by task tuple."""
    parallel = Parallel(n_jobs=jobs, prefer="processes")
    results = parallel(
        delayed(fit_task)(geo, origin, receiver, horizon, growth_by_geo[geo])
        for geo, origin, receiver, horizon in tasks
    )
    return [result for result in results if result is not None]


def write_results(results: list[dict], out_dir: str) -> pd.DataFrame:
    """Write the tidy results and the beta matrix, returning the tidy frame."""
    frame = pd.DataFrame(results).sort_values(
        ["geo", "origin_component", "receiving_component", "horizon"]
    )
    frame = frame.reset_index(drop=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path / LONG_FILE, index=False)

    beta = frame.pivot(
        index=["geo", "origin_component", "receiving_component"],
        columns="horizon",
        values="beta",
    )
    beta = beta[list(HORIZONS)]
    beta.columns = [f"horizon_{horizon}" for horizon in HORIZONS]
    beta.to_csv(out_path / BETA_FILE)
    return frame


def validate_results(frame: pd.DataFrame) -> None:
    """Check exclusions, uniqueness, and finite impulse responses."""
    assert not frame.empty, "No local projections were fitted"
    assert not frame["receiving_component"].isin(EXCLUDED_RECEIVERS).any()
    assert (frame["receiving_component"] != frame["origin_component"]).all()
    assert frame["origin_component"].isin(ORIGINS).all()
    assert np.isfinite(frame["beta"]).all()
    keys = ["geo", "origin_component", "receiving_component", "horizon"]
    assert not frame.duplicated(keys).any()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit HICP local projections for energy shock origins."
    )
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--out-dir", default=".", help="Directory for the output CSV files."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes (default: -1).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the local projection pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    growth_by_geo = load_growth(args.db)
    tasks = build_tasks(growth_by_geo)
    logging.info("Fitting %d local projections", len(tasks))

    results = fit_all(tasks, growth_by_geo, args.jobs)
    frame = write_results(results, args.out_dir)
    validate_results(frame)

    logging.info(
        "Wrote %d rows across %d geographies",
        len(frame),
        frame["geo"].nunique(),
    )
    logging.info("Horizon counts: %s", frame["horizon"].value_counts().to_dict())
    logging.info("Files written to %s", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()

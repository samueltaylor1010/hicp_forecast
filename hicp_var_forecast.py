"""Country-level Lasso VAR forecasts for HICP components.

Each country is modelled independently using its own retained COICOP
components. Features are current and lagged monthly log differences; the
three separate VARs forecast cumulative log inflation over 1, 2, and 3
months. The model is tuned with expanding forward validation and the
one-standard-error rule using MAE.
"""

import argparse
import logging
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import StandardScaler

DB_PATH = "hicp.db"
OUTPUT_PATH = "hicp_var_models.joblib"
EXCLUDED_COMPONENTS = ("TOTAL", "CP04", "CP07")
EXCLUDED_GEOS = ("EA",)
HORIZONS = (1, 2, 3)
MAX_LAG = 3
MIN_TRAIN_MONTHS = 60
VALIDATION_STEP_MONTHS = 3
ALPHAS = np.logspace(-1, -6, 21)
PATH_MAX_ITER = 20_000
PATH_TOL = 1e-3
FINAL_MAX_ITER = 50_000
FINAL_TOL = 1e-5


def load_panel(db_path):
    """Load and validate the HICP component panel."""
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT time_period, geo, coicop, hicp_index
            FROM hicp_panel_data
            """,
            conn,
            parse_dates=["time_period"],
        )

    frame = frame[
        ~frame["coicop"].isin(EXCLUDED_COMPONENTS)
        & ~frame["geo"].isin(EXCLUDED_GEOS)
    ].copy()
    if frame.empty:
        raise ValueError("No observations remain after applying exclusions")
    if (frame["hicp_index"] <= 0).any():
        raise ValueError("HICP indices must be strictly positive")
    return frame


def country_matrices(frame, geo, components):
    """Build monthly-difference features and cumulative targets for a country."""
    country = frame[frame["geo"] == geo]
    panel = country.pivot(
        index="time_period", columns="coicop", values="hicp_index"
    ).sort_index()

    missing = set(components) - set(panel.columns)
    if missing:
        raise ValueError(f"{geo} is missing components: {sorted(missing)}")

    panel = panel[components]
    calendar = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    panel = panel.reindex(calendar)
    log_index = np.log(panel)
    monthly_change = log_index.diff()

    feature_blocks = []
    for component in components:
        for lag in range(MAX_LAG + 1):
            name = f"{component}_lag{lag}"
            feature_blocks.append(monthly_change[component].shift(lag).rename(name))

    features = pd.concat(feature_blocks, axis=1)
    features = features.loc[features.notna().all(axis=1)]
    if features.empty:
        raise ValueError(f"{geo} has no complete feature rows")

    targets = {}
    for horizon in HORIZONS:
        target = log_index.shift(-horizon).sub(log_index)
        targets[horizon] = target.reindex(features.index)

    return features, targets


def make_folds(dates, valid_targets, horizon):
    """Create expanding-window folds with an h-month purge before validation."""
    folds = []
    for validation_idx in np.flatnonzero(valid_targets):
        cutoff = dates[validation_idx] - pd.DateOffset(months=horizon)
        train_end = dates.searchsorted(cutoff, side="right") - 1
        if train_end + 1 >= MIN_TRAIN_MONTHS:
            folds.append((train_end, validation_idx))
    return folds[::VALIDATION_STEP_MONTHS]


def _select_alpha(mae, mae_se):
    """Apply the one-standard-error rule and return the most-penalized alpha."""
    best = int(np.argmin(mae))
    threshold = mae[best] + mae_se[best]
    eligible = np.flatnonzero(mae <= threshold + 1e-12)
    return int(eligible[0])


def fit_country_horizon(
    geo,
    horizon,
    features,
    dates,
    targets,
    folds,
    components,
    feature_names,
):
    """Tune and fit all component equations for one country and horizon."""
    n_folds = len(folds)
    n_components = len(components)
    n_alphas = len(ALPHAS)
    absolute_errors = np.empty((n_folds, n_components, n_alphas))
    converged = np.zeros((n_components, n_folds), dtype=bool)

    for fold_idx, (train_end, validation_idx) in enumerate(folds):
        train_rows = np.arange(train_end + 1)
        scaler = StandardScaler().fit(features[train_rows])
        x_train = scaler.transform(features[train_rows])
        x_validation = scaler.transform(features[validation_idx : validation_idx + 1])

        for component_idx in range(n_components):
            y_train = targets[train_rows, component_idx]
            y_mean = y_train.mean()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                path_alphas, coefficients, _, n_iters = lasso_path(
                    x_train,
                    y_train - y_mean,
                    alphas=ALPHAS,
                    max_iter=PATH_MAX_ITER,
                    tol=PATH_TOL,
                    return_n_iter=True,
                )
            if not np.allclose(path_alphas, ALPHAS):
                raise RuntimeError("Unexpected lasso path alpha ordering")
            converged[component_idx, fold_idx] = np.any(
                np.asarray(n_iters) < PATH_MAX_ITER
            )
            predictions = y_mean + x_validation @ coefficients
            absolute_errors[fold_idx, component_idx] = np.abs(
                predictions[0] - targets[validation_idx, component_idx]
            )

    failed = np.flatnonzero(~converged.any(axis=1))
    if failed.size:
        names = ", ".join(components[index] for index in failed)
        raise RuntimeError(
            f"{geo} horizon {horizon}: no tested alpha converged for {names}"
        )

    mae = absolute_errors.mean(axis=0)
    mae_se = absolute_errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    selected_indices = np.array(
        [_select_alpha(mae[i], mae_se[i]) for i in range(n_components)]
    )
    selected_alphas = ALPHAS[selected_indices]

    coefficient_matrix = np.empty((n_components, len(feature_names)))
    intercepts = np.empty(n_components)
    forecasts = np.empty(n_components)
    nonzero = np.empty(n_components, dtype=int)
    train_counts = np.empty(n_components, dtype=int)
    train_starts = []
    train_ends = []

    for component_idx, alpha in enumerate(selected_alphas):
        valid_rows = np.flatnonzero(np.isfinite(targets[:, component_idx]))
        scaler = StandardScaler().fit(features[valid_rows])
        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=FINAL_MAX_ITER,
            tol=FINAL_TOL,
            selection="cyclic",
        ).fit(scaler.transform(features[valid_rows]), targets[valid_rows, component_idx])

        scaled_coef = model.coef_
        raw_coef = scaled_coef / scaler.scale_
        raw_intercept = model.intercept_ - np.dot(
            scaler.mean_ / scaler.scale_, scaled_coef
        )
        coefficient_matrix[component_idx] = raw_coef
        intercepts[component_idx] = raw_intercept
        forecasts[component_idx] = raw_intercept + features[-1] @ raw_coef
        nonzero[component_idx] = np.count_nonzero(raw_coef)
        train_counts[component_idx] = len(valid_rows)
        train_starts.append(pd.Timestamp(dates[valid_rows[0]]).date().isoformat())
        train_ends.append(pd.Timestamp(dates[valid_rows[-1]]).date().isoformat())

    return {
        "geo": geo,
        "horizon": horizon,
        "components": list(components),
        "features": list(feature_names),
        "coefficients": coefficient_matrix,
        "intercepts": intercepts,
        "alphas": selected_alphas,
        "mae": mae,
        "mae_se": mae_se,
        "nonzero": nonzero,
        "n_train": train_counts,
        "train_start": train_starts,
        "train_end": train_ends,
        "forecast": forecasts,
        "n_folds": n_folds,
    }


def build_tasks(frame, geos, components):
    """Build independent country-horizon fitting tasks."""
    tasks = []
    metadata = {}
    for geo in geos:
        features, targets = country_matrices(frame, geo, components)
        feature_names = list(features.columns)
        dates = features.index
        metadata[geo] = {
            "feature_start": dates[0].date().isoformat(),
            "feature_end": dates[-1].date().isoformat(),
            "feature_rows": len(features),
            "features": feature_names,
            "latest_features": pd.Series(
                features.iloc[-1].to_numpy(), index=feature_names
            ),
        }

        for horizon in HORIZONS:
            target_frame = targets[horizon]
            valid_targets = target_frame.notna().all(axis=1).to_numpy()
            folds = make_folds(dates, valid_targets, horizon)
            if not folds:
                raise ValueError(f"{geo} horizon {horizon} has no validation folds")
            tasks.append(
                (
                    geo,
                    horizon,
                    features.to_numpy(dtype=float),
                    dates.to_numpy(),
                    target_frame.to_numpy(dtype=float),
                    folds,
                    components,
                    feature_names,
                )
            )
    return tasks, metadata


def assemble_results(results, metadata, components):
    """Convert task outputs into an auditable, dashboard-friendly bundle."""
    countries = {}
    forecast_rows = []

    for result in results:
        geo = result["geo"]
        horizon = result["horizon"]
        feature_names = result["features"]
        components_h = result["components"]
        origin = pd.Timestamp(metadata[geo]["feature_end"])
        target_period = origin + pd.DateOffset(months=horizon)

        country = countries.setdefault(
            geo,
            {
                "geo": geo,
                "components": components_h,
                "features": feature_names,
                "feature_start": metadata[geo]["feature_start"],
                "feature_end": metadata[geo]["feature_end"],
                "latest_features": metadata[geo]["latest_features"],
                "horizons": {},
            },
        )

        horizon_result = {
            "horizon": horizon,
            "target": f"log(P[t+{horizon}] / P[t])",
            "components": components_h,
            "features": feature_names,
            "coefficients": pd.DataFrame(
                result["coefficients"], index=components_h, columns=feature_names
            ),
            "intercepts": pd.Series(result["intercepts"], index=components_h),
            "alphas": pd.Series(result["alphas"], index=components_h),
            "cv_mae": pd.DataFrame(
                result["mae"], index=components_h, columns=ALPHAS
            ),
            "cv_mae_se": pd.DataFrame(
                result["mae_se"], index=components_h, columns=ALPHAS
            ),
            "nonzero": pd.Series(result["nonzero"], index=components_h),
            "n_train": pd.Series(result["n_train"], index=components_h),
            "train_start": pd.Series(result["train_start"], index=components_h),
            "train_end": pd.Series(result["train_end"], index=components_h),
            "n_folds": result["n_folds"],
            "forecast_origin": origin,
            "forecast_period": target_period,
            "forecast_log": pd.Series(result["forecast"], index=components_h),
            "forecast_pct": pd.Series(
                100.0 * np.expm1(result["forecast"]), index=components_h
            ),
        }
        country["horizons"][horizon] = horizon_result

        for component_idx, component in enumerate(components_h):
            forecast_rows.append(
                {
                    "geo": geo,
                    "coicop": component,
                    "horizon": horizon,
                    "forecast_origin": origin,
                    "forecast_period": target_period,
                    "forecast_log": result["forecast"][component_idx],
                    "forecast_pct": 100.0
                    * np.expm1(result["forecast"][component_idx]),
                }
            )

    forecasts = pd.DataFrame(forecast_rows)
    return countries, forecasts


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="Path to hicp.db")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Artifact path")
    parser.add_argument(
        "--jobs", type=int, default=-1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--geos",
        help="Optional comma-separated country codes for a smoke test",
    )
    return parser.parse_args()


def main():
    """Fit all country VARs and serialize the model bundle."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    frame = load_panel(args.db)
    components = sorted(frame["coicop"].unique())
    geos = sorted(frame["geo"].unique())
    if args.geos:
        requested = [value.strip() for value in args.geos.split(",") if value.strip()]
        geos = [geo for geo in requested if geo in geos]
        if not geos:
            raise ValueError("No requested geographies are available")

    logging.info(
        "Fitting %d countries x %d horizons x %d components",
        len(geos),
        len(HORIZONS),
        len(components),
    )
    tasks, metadata = build_tasks(frame, geos, components)
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_country_horizon)(*task) for task in tasks
    )
    countries, forecasts = assemble_results(results, metadata, components)

    bundle = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(Path(args.db).resolve()),
            "countries": geos,
            "components": components,
            "excluded_components": list(EXCLUDED_COMPONENTS),
            "excluded_geos": list(EXCLUDED_GEOS),
            "horizons": list(HORIZONS),
            "max_lag": MAX_LAG,
            "features_per_country": len(components) * (MAX_LAG + 1),
            "feature_transform": "monthly log difference: log(P[t] / P[t-1])",
            "target_transform": "cumulative log change: log(P[t+h] / P[t])",
            "validation": "expanding window, quarterly origins, h-month purge",
            "selection": "Lasso, 1-SE rule on MAE",
            "alpha_grid": ALPHAS.tolist(),
        },
        "countries": countries,
        "forecasts": forecasts,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    logging.info(
        "Saved %d country models and %d forecasts to %s",
        len(countries),
        len(forecasts),
        output,
    )


if __name__ == "__main__":
    main()

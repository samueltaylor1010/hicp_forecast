"""Country-level dynamic ridge VAR for conditional HICP shock scenarios.

Each country is modelled independently as a VAR(4) in monthly log
differences. Ridge regression retains cross-component channels that the
sparse direct-horizon Lasso model removed. The fitted system is iterated
forward to produce three-month baseline and shock-response paths.
"""

import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = "hicp.db"
OUTPUT_PATH = "hicp_var_dynamic.joblib"
EXCLUDED_COMPONENTS = ("TOTAL", "CP04", "CP07")
EXCLUDED_GEOS = ("EA",)
N_LAGS = 4
HORIZON = 3
MIN_TRAIN_MONTHS = 60
VALIDATION_STEP_MONTHS = 3
ALPHAS = np.logspace(-4, 3, 22)
MONTH_DUMMIES = tuple(range(2, 13))


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


def build_country_data(frame, geo, components):
    """Build lagged monthly differences, targets, dummies, and latest state."""
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
    changes = np.log(panel).diff()

    feature_blocks = []
    for component in components:
        for lag in range(N_LAGS):
            feature_blocks.append(
                changes[component].shift(lag).rename(f"{component}_lag{lag}")
            )
    lagged_features = pd.concat(feature_blocks, axis=1)
    lagged_features = lagged_features.loc[lagged_features.notna().all(axis=1)]
    if lagged_features.empty:
        raise ValueError(f"{geo} has no complete feature rows")

    target = changes.shift(-1).reindex(lagged_features.index)
    target_month = (lagged_features.index + pd.DateOffset(months=1)).month
    dummies = pd.DataFrame(
        {
            f"month_{month}": (target_month == month).astype(float)
            for month in MONTH_DUMMIES
        },
        index=lagged_features.index,
    )
    features = pd.concat([lagged_features, dummies], axis=1)

    latest_origin = features.index[-1]
    state = changes.loc[:latest_origin].tail(N_LAGS).iloc[::-1]
    if state.shape != (N_LAGS, len(components)) or state.isna().any().any():
        raise ValueError(f"{geo} does not have a complete latest state")

    return features, target, state


def make_folds(dates, valid_targets):
    """Create expanding-window folds with a one-month target purge."""
    folds = []
    for validation_idx in np.flatnonzero(valid_targets):
        cutoff = dates[validation_idx] - pd.DateOffset(months=1)
        train_end = dates.searchsorted(cutoff, side="right") - 1
        if train_end + 1 >= MIN_TRAIN_MONTHS:
            folds.append((train_end, validation_idx))
    return folds[::VALIDATION_STEP_MONTHS]


def build_transition(coefficients, components, feature_names):
    """Convert a raw coefficient matrix into VAR transition and seasonal terms."""
    coefficient_frame = pd.DataFrame(
        coefficients, index=components, columns=feature_names
    )
    transition = np.empty((N_LAGS, len(components), len(components)))
    for lag in range(N_LAGS):
        for source_idx, source in enumerate(components):
            transition[lag, :, source_idx] = coefficient_frame[
                f"{source}_lag{lag}"
            ].to_numpy()

    month_effects = np.zeros((12, len(components)))
    for month in MONTH_DUMMIES:
        month_effects[month - 1] = coefficient_frame[f"month_{month}"].to_numpy()

    return coefficient_frame, transition, month_effects


def companion_eigenvalue(transition):
    """Return the largest absolute eigenvalue of the VAR companion matrix."""
    n_components = transition.shape[1]
    top = np.hstack([transition[lag] for lag in range(N_LAGS)])
    bottom = np.vstack(
        [
            np.hstack(
                [
                    np.eye(n_components)
                    if row == col
                    else np.zeros((n_components, n_components))
                    for col in range(N_LAGS)
                ]
            )
            for row in range(N_LAGS - 1)
        ]
    )
    companion = np.vstack([top, bottom])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def fit_final_equations(features, targets, alphas, components):
    """Fit one Ridge equation per component with the selected alpha."""
    valid_rows = np.flatnonzero(np.isfinite(targets).all(axis=1))
    scaler = StandardScaler().fit(features[valid_rows])
    x_train = scaler.transform(features[valid_rows])

    coefficients = np.empty((len(components), features.shape[1]))
    intercepts = np.empty(len(components))
    residuals = np.empty((len(valid_rows), len(components)))

    for component_idx, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha, fit_intercept=True).fit(
            x_train, targets[valid_rows, component_idx]
        )
        scaled_coef = model.coef_
        coefficients[component_idx] = scaled_coef / scaler.scale_
        intercepts[component_idx] = model.intercept_ - np.dot(
            scaler.mean_ / scaler.scale_, scaled_coef
        )
        residuals[:, component_idx] = (
            targets[valid_rows, component_idx]
            - intercepts[component_idx]
            - features[valid_rows] @ coefficients[component_idx]
        )

    return coefficients, intercepts, residuals, valid_rows


def simulate_path(
    transition,
    intercepts,
    month_effects,
    initial_state,
    origin,
    horizon=HORIZON,
    shocked_component=None,
    shock_size=0.0,
):
    """Iterate a fitted VAR forward, optionally shocking current z."""
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (N_LAGS, len(intercepts)):
        raise ValueError("Initial state has the wrong shape")
    if shocked_component is not None:
        state[0, shocked_component] += shock_size

    path = []
    for step in range(1, horizon + 1):
        target_date = pd.Timestamp(origin) + pd.DateOffset(months=step)
        next_value = intercepts + month_effects[target_date.month - 1]
        for lag in range(N_LAGS):
            next_value = next_value + transition[lag] @ state[lag]
        path.append(next_value)
        state = np.vstack([next_value, state[:-1]])
    return np.asarray(path)


def fit_country(geo, features, target, state, folds, components, feature_names):
    """Tune, fit, stabilize, and simulate one country VAR."""
    x_all = features.to_numpy(dtype=float)
    y_all = target.to_numpy(dtype=float)
    n_components = len(components)
    absolute_errors = np.empty((len(folds), n_components, len(ALPHAS)))

    for fold_idx, (train_end, validation_idx) in enumerate(folds):
        train_rows = np.arange(train_end + 1)
        scaler = StandardScaler().fit(x_all[train_rows])
        x_train = scaler.transform(x_all[train_rows])
        x_validation = scaler.transform(x_all[validation_idx : validation_idx + 1])

        for alpha_idx, alpha in enumerate(ALPHAS):
            model = Ridge(alpha=alpha, fit_intercept=True).fit(
                x_train, y_all[train_rows]
            )
            predictions = model.predict(x_validation)
            absolute_errors[fold_idx, :, alpha_idx] = np.abs(
                predictions[0] - y_all[validation_idx]
            )

    mae = absolute_errors.mean(axis=0)
    mae_se = absolute_errors.std(axis=0, ddof=1) / np.sqrt(len(folds))
    mae_optimal = np.argmin(mae, axis=1)
    thresholds = mae[np.arange(n_components), mae_optimal] + mae_se[
        np.arange(n_components), mae_optimal
    ]
    selected_indices = np.array(
        [
            np.flatnonzero(mae[i] <= thresholds[i] + 1e-12)[0]
            for i in range(n_components)
        ]
    )
    used_indices = selected_indices.copy()
    adjustment_steps = 0

    for _ in range(len(ALPHAS)):
        used_alphas = ALPHAS[used_indices]
        coefficients, intercepts, residuals, valid_rows = fit_final_equations(
            x_all, y_all, used_alphas, components
        )
        coefficient_frame, transition, month_effects = build_transition(
            coefficients, components, feature_names
        )
        max_eigenvalue = companion_eigenvalue(transition)
        if max_eigenvalue < 1.0:
            break
        if np.all(used_indices >= len(ALPHAS) - 1):
            raise RuntimeError(f"{geo} could not be stabilized with Ridge")
        used_indices = np.minimum(used_indices + 1, len(ALPHAS) - 1)
        adjustment_steps += 1
    else:
        raise RuntimeError(f"{geo} failed the VAR stability check")

    latest_origin = features.index[-1]
    baseline = simulate_path(
        transition,
        intercepts,
        month_effects,
        state.to_numpy(),
        latest_origin,
    )
    residual_covariance = np.cov(residuals, rowvar=False, ddof=1)

    return {
        "geo": geo,
        "components": list(components),
        "features": list(feature_names),
        "coefficients": coefficient_frame.to_numpy(),
        "intercepts": intercepts,
        "transition": transition,
        "month_effects": month_effects,
        "alphas": ALPHAS[used_indices],
        "cv_alphas": ALPHAS[selected_indices],
        "mae": mae,
        "mae_se": mae_se,
        "residual_covariance": residual_covariance,
        "n_train": len(valid_rows),
        "train_start": pd.Timestamp(features.index[valid_rows[0]]).date().isoformat(),
        "train_end": pd.Timestamp(features.index[valid_rows[-1]]).date().isoformat(),
        "n_folds": len(folds),
        "max_eigenvalue": max_eigenvalue,
        "stability_adjustments": adjustment_steps,
        "state": state,
        "latest_origin": latest_origin,
        "baseline": baseline,
    }


def build_tasks(frame, geos, components):
    """Build country fitting tasks and baseline forecast rows."""
    tasks = []
    for geo in geos:
        features, target, state = build_country_data(frame, geo, components)
        valid_targets = target.notna().all(axis=1).to_numpy()
        folds = make_folds(features.index, valid_targets)
        if not folds:
            raise ValueError(f"{geo} has no validation folds")
        tasks.append(
            (
                geo,
                features,
                target,
                state,
                folds,
                components,
                list(features.columns),
            )
        )
    return tasks


def assemble_bundle(results, metadata):
    """Convert country fits into the serialized model bundle."""
    countries = {}
    baseline_rows = []

    for result in results:
        geo = result["geo"]
        components = result["components"]
        origin = pd.Timestamp(result["latest_origin"])
        state = result["state"]
        state_dates = pd.date_range(end=origin, periods=N_LAGS, freq="MS")[::-1]

        baseline = result["baseline"]
        baseline_frame = pd.DataFrame(
            baseline,
            index=pd.date_range(
                start=origin + pd.DateOffset(months=1),
                periods=HORIZON,
                freq="MS",
            ),
            columns=components,
        )
        month_effects = pd.DataFrame(
            result["month_effects"],
            index=range(1, 13),
            columns=components,
        )
        residual_covariance = pd.DataFrame(
            result["residual_covariance"], index=components, columns=components
        )

        countries[geo] = {
            "geo": geo,
            "components": components,
            "features": result["features"],
            "coefficients": pd.DataFrame(
                result["coefficients"], index=components, columns=result["features"]
            ),
            "intercepts": pd.Series(result["intercepts"], index=components),
            "transition": result["transition"],
            "month_effects": month_effects,
            "alphas": pd.Series(result["alphas"], index=components),
            "cv_alphas": pd.Series(result["cv_alphas"], index=components),
            "cv_mae": pd.DataFrame(
                result["mae"], index=components, columns=ALPHAS
            ),
            "cv_mae_se": pd.DataFrame(
                result["mae_se"], index=components, columns=ALPHAS
            ),
            "residual_covariance": residual_covariance,
            "n_train": result["n_train"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "n_folds": result["n_folds"],
            "max_eigenvalue": result["max_eigenvalue"],
            "stable": result["max_eigenvalue"] < 1.0,
            "stability_adjustments": result["stability_adjustments"],
            "state": pd.DataFrame(
                state.to_numpy(), index=state_dates, columns=components
            ),
            "latest_origin": origin,
            "baseline": baseline_frame,
        }

        for horizon_idx, period in enumerate(baseline_frame.index, start=1):
            for component_idx, component in enumerate(components):
                value = baseline_frame.iloc[horizon_idx - 1, component_idx]
                baseline_rows.append(
                    {
                        "geo": geo,
                        "coicop": component,
                        "horizon": horizon_idx,
                        "forecast_origin": origin,
                        "forecast_period": period,
                        "forecast_log": value,
                        "forecast_pct": 100.0 * np.expm1(value),
                    }
                )

    bundle = {
        "metadata": {
            **metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": sorted({c for result in results for c in result["components"]}),
            "n_lags": N_LAGS,
            "horizon": HORIZON,
            "estimator": "Ridge VAR, one equation per component",
            "selection": "smallest alpha within one standard error of minimum MAE",
            "seasonality": "target-month dummies, January baseline",
            "shock_semantics": "conditional scenario holding other current components fixed",
            "alpha_grid": ALPHAS.tolist(),
        },
        "countries": countries,
        "baseline_forecasts": pd.DataFrame(baseline_rows),
    }
    return bundle


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
    """Fit all country VARs and serialize the dynamic model bundle."""
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

    logging.info("Fitting %d country VARs with %d components", len(geos), len(components))
    tasks = build_tasks(frame, geos, components)
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_country)(*task) for task in tasks
    )
    bundle = assemble_bundle(
        results,
        {
            "db_path": str(Path(args.db).resolve()),
            "countries": geos,
            "excluded_components": list(EXCLUDED_COMPONENTS),
            "excluded_geos": list(EXCLUDED_GEOS),
            "validation": "expanding window, quarterly origins, one-month purge",
        },
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    logging.info("Saved %d country VARs to %s", len(bundle["countries"]), output)


if __name__ == "__main__":
    main()

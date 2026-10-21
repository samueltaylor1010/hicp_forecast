"""Run a conditional shock scenario through the dynamic country VAR."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hicp_var_dynamic import HORIZON, simulate_path

ARTIFACT_PATH = "hicp_var_dynamic.joblib"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", required=True, help="Geography code, e.g. AT")
    parser.add_argument(
        "--source", required=True, help="Component receiving the shock, e.g. CP045"
    )
    parser.add_argument(
        "--shock",
        type=float,
        default=0.01,
        help="Shock in current monthly log points; 0.01 = +1 percentage point",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help="Simulation horizon in months",
    )
    parser.add_argument(
        "--artifact", default=ARTIFACT_PATH, help="Path to dynamic VAR artifact"
    )
    parser.add_argument("--output", help="Output CSV path")
    return parser.parse_args()


def main():
    """Simulate baseline and shocked paths and save the response table."""
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("Horizon must be positive")
    if not np.isfinite(args.shock):
        raise ValueError("Shock must be finite")

    bundle = joblib.load(args.artifact)
    country = bundle["countries"].get(args.geo.upper())
    if country is None:
        raise ValueError(f"Geography {args.geo.upper()} is not in the artifact")

    components = country["components"]
    source = args.source.upper()
    if source not in components:
        raise ValueError(f"Component {source} is not available for {args.geo.upper()}")

    source_idx = components.index(source)
    origin = pd.Timestamp(country["latest_origin"])
    state = country["state"].to_numpy(dtype=float)
    transition = country["transition"]
    intercepts = country["intercepts"].to_numpy(dtype=float)
    month_effects = country["month_effects"].to_numpy(dtype=float)

    baseline = simulate_path(
        transition,
        intercepts,
        month_effects,
        state,
        origin,
        horizon=args.horizon,
    )
    shocked = simulate_path(
        transition,
        intercepts,
        month_effects,
        state,
        origin,
        horizon=args.horizon,
        shocked_component=source_idx,
        shock_size=args.shock,
    )

    monthly_response = shocked - baseline
    cumulative_response = np.cumsum(monthly_response, axis=0)
    cumulative_pct = 100.0 * np.expm1(cumulative_response)
    periods = pd.date_range(
        start=origin + pd.DateOffset(months=1),
        periods=args.horizon,
        freq="MS",
    )

    rows = []
    for horizon_idx in range(args.horizon):
        for component_idx, component in enumerate(components):
            rows.append(
                {
                    "geo": country["geo"],
                    "source": source,
                    "shock_log": args.shock,
                    "coicop": component,
                    "horizon": horizon_idx + 1,
                    "forecast_period": periods[horizon_idx],
                    "baseline_monthly_log": baseline[horizon_idx, component_idx],
                    "shocked_monthly_log": shocked[horizon_idx, component_idx],
                    "monthly_response_log": monthly_response[horizon_idx, component_idx],
                    "cumulative_response_log": cumulative_response[
                        horizon_idx, component_idx
                    ],
                    "cumulative_response_pct": cumulative_pct[
                        horizon_idx, component_idx
                    ],
                }
            )

    results = pd.DataFrame(rows)
    output = Path(
        args.output or f"shock_{country['geo']}_{source}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = results.pivot(
        index="coicop", columns="horizon", values="cumulative_response_pct"
    )
    print(summary.round(4).to_string())
    print(f"\nSaved response table to {output}")


if __name__ == "__main__":
    main()

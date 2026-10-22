"""Scenario and EA aggregation engine for the dashboard.

The dashboard shock is a percent change to the latest origin index of CP045
and CP0722. The resulting log shock is propagated to the target period with
the fitted direct-horizon pass-through coefficients. All EA and country
aggregates are bottom-up using HICP country and item weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SHOCK_COICOPS = ("CP045", "CP0722")
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
EA_GEO = "EA"
EPSILON = 1e-12


@dataclass(frozen=True)
class DashboardData:
    baselines: pd.DataFrame
    multipliers: pd.DataFrame
    history: pd.DataFrame
    weights: pd.DataFrame
    headline_history: pd.DataFrame
    ea_component_history: pd.DataFrame
    countries: list[str]
    geos: list[str]
    coicops: list[str]
    origin: pd.Timestamp
    target_period: pd.Timestamp
    weight_year: int
    baseline_wide: pd.DataFrame
    lag0_wide: pd.DataFrame
    beta_sum_wide: pd.DataFrame
    item_weights_wide: pd.DataFrame
    item_norm_wide: pd.DataFrame
    country_weights: pd.Series
    country_norm: pd.Series


@dataclass(frozen=True)
class AggregationResult:
    country_headlines: pd.DataFrame
    ea_headline: dict[str, float]
    country_components: pd.DataFrame
    ea_components: pd.DataFrame
    ea_countries: pd.DataFrame


@dataclass(frozen=True)
class GraphSeries:
    code: str
    history: pd.DataFrame
    origin_index: float
    baseline: float
    scenario: float
    is_source: bool
    shock_percent: float | None


def load_dashboard_data(data_dir: str | Path) -> DashboardData:
    data_dir = Path(data_dir)
    baselines = pd.read_csv(
        data_dir / "baselines.csv",
        parse_dates=["forecast_origin", "target_period"],
    )
    multipliers = pd.read_csv(data_dir / "multipliers.csv")
    history = pd.read_csv(data_dir / "historical_indices.csv", parse_dates=["date"])
    weights = pd.read_csv(data_dir / "weights.csv")
    headline_history = pd.read_csv(
        data_dir / "headline_history.csv", parse_dates=["date"]
    )
    ea_component_history = pd.read_csv(
        data_dir / "ea_component_history.csv", parse_dates=["date"]
    )

    _validate_frames(
        baselines,
        multipliers,
        history,
        weights,
        headline_history,
        ea_component_history,
    )

    countries = sorted(baselines["country"].unique())
    geos = [EA_GEO, *countries]
    coicops = [code for code in RETAINED_COICOPS if code in set(baselines["coicop"])]
    target_period = pd.Timestamp(baselines["target_period"].iloc[0])
    origin = pd.Timestamp(baselines["forecast_origin"].iloc[0])
    weight_year = int(weights["weight_year"].iloc[0])

    baseline_wide = (
        baselines.pivot(
            index="country", columns="coicop", values="baseline_index_t3"
        )
        .reindex(index=countries, columns=coicops)
        .astype(float)
    )
    lag0_wide = (
        baselines.pivot(
            index="country", columns="coicop", values="lag0_coefficient"
        )
        .reindex(index=countries, columns=coicops)
        .astype(float)
    )

    beta_045 = (
        multipliers[multipliers["shock_variable"] == "CP045"]
        .pivot(
            index="country",
            columns="target_coicop",
            values="beta_multiplier",
        )
        .reindex(index=countries, columns=coicops)
        .astype(float)
    )
    beta_0722 = (
        multipliers[multipliers["shock_variable"] == "CP0722"]
        .pivot(
            index="country",
            columns="target_coicop",
            values="beta_multiplier",
        )
        .reindex(index=countries, columns=coicops)
        .astype(float)
    )
    beta_sum_wide = beta_045.add(beta_0722, fill_value=0.0)

    item_weights_wide = (
        weights.pivot(index="country", columns="coicop", values="item_weight")
        .reindex(index=countries, columns=coicops)
        .astype(float)
    )
    item_norm_wide = item_weights_wide.div(
        item_weights_wide.sum(axis=1), axis=0
    )
    country_weights = (
        weights.groupby("country")["country_weight"]
        .first()
        .reindex(countries)
        .astype(float)
    )
    country_norm = country_weights / country_weights.sum()

    return DashboardData(
        baselines=baselines,
        multipliers=multipliers,
        history=history,
        weights=weights,
        headline_history=headline_history,
        ea_component_history=ea_component_history,
        countries=countries,
        geos=geos,
        coicops=coicops,
        origin=origin,
        target_period=target_period,
        weight_year=weight_year,
        baseline_wide=baseline_wide,
        lag0_wide=lag0_wide,
        beta_sum_wide=beta_sum_wide,
        item_weights_wide=item_weights_wide,
        item_norm_wide=item_norm_wide,
        country_weights=country_weights,
        country_norm=country_norm,
    )


def _validate_frames(
    baselines: pd.DataFrame,
    multipliers: pd.DataFrame,
    history: pd.DataFrame,
    weights: pd.DataFrame,
    headline_history: pd.DataFrame,
    ea_component_history: pd.DataFrame,
) -> None:
    if baselines.empty:
        raise ValueError("Empty baselines")
    if baselines["target_period"].nunique() != 1:
        raise ValueError("Expected exactly one target period")
    if baselines["forecast_origin"].nunique() != 1:
        raise ValueError("Expected exactly one forecast origin")
    expected_origin = pd.Timestamp(
        baselines["target_period"].iloc[0]
    ) - pd.DateOffset(months=3)
    if pd.Timestamp(baselines["forecast_origin"].iloc[0]) != expected_origin:
        raise ValueError("Forecast origin is not target period minus 3 months")
    if baselines.duplicated(["country", "coicop"]).any():
        raise ValueError("Duplicate baseline keys")
    if not np.isfinite(
        baselines[["baseline_index_t3", "lag0_coefficient"]]
    ).all().all():
        raise ValueError("Non-finite baseline values")

    if multipliers.duplicated(
        ["country", "target_coicop", "shock_variable"]
    ).any():
        raise ValueError("Duplicate multiplier keys")
    if not np.isfinite(multipliers["beta_multiplier"]).all():
        raise ValueError("Non-finite multipliers")

    if history.duplicated(["country", "coicop", "date"]).any():
        raise ValueError("Duplicate history keys")
    if not np.isfinite(history["index_value"]).all():
        raise ValueError("Non-finite history values")
    if history["date"].max() != pd.Timestamp(
        baselines["forecast_origin"].iloc[0]
    ):
        raise ValueError("History does not end at the forecast origin")

    if weights.duplicated(["country", "coicop"]).any():
        raise ValueError("Duplicate weight keys")
    if not np.isfinite(
        weights[["item_weight", "country_weight"]]
    ).all().all():
        raise ValueError("Non-finite weights")
    if (weights[["item_weight", "country_weight"]] <= 0).any().any():
        raise ValueError("Weights must be positive")
    if weights["country"].nunique() != 21:
        raise ValueError("Expected 21 countries in weights")
    if weights["coicop"].nunique() != len(RETAINED_COICOPS):
        raise ValueError("Expected 15 components in weights")

    if headline_history.duplicated(["country", "date"]).any():
        raise ValueError("Duplicate headline history keys")
    if not np.isfinite(headline_history["index_value"]).all():
        raise ValueError("Non-finite headline history values")
    if EA_GEO not in set(headline_history["country"]):
        raise ValueError("Headline history is missing EA")

    if ea_component_history.duplicated(["coicop", "date"]).any():
        raise ValueError("Duplicate EA component history keys")
    if not np.isfinite(ea_component_history["index_value"]).all():
        raise ValueError("Non-finite EA component history values")


def percent_to_log_shock(percent: float) -> float:
    """Convert a percent change in the latest index to a log shock."""
    if not np.isfinite(percent) or percent <= -100.0:
        raise ValueError("Shock percent must be finite and greater than -100")
    return float(np.log1p(percent / 100.0))


def scenario_components(
    data: DashboardData,
    shock_percent_by_country: dict[str, float],
) -> pd.DataFrame:
    """Apply origin-index shocks and return target component scenarios.

    The shock is applied to the latest origin index of CP045 and CP0722. Its
    log value is propagated to the target period with lag0 for the shock
    sources and with the sum of the two pass-through betas for targets.
    """
    percentages = np.array(
        [
            float(shock_percent_by_country.get(country, 0.0))
            for country in data.countries
        ],
        dtype=float,
    )
    if not np.isfinite(percentages).all():
        raise ValueError("Shock percentages must be finite")
    if (percentages <= -100.0).any():
        raise ValueError("Shock percentages must be greater than -100")

    log_shocks = np.log1p(percentages / 100.0)[:, None]
    baseline = data.baseline_wide.to_numpy(dtype=float)
    beta_sum = data.beta_sum_wide.to_numpy(dtype=float)
    lag0 = data.lag0_wide.to_numpy(dtype=float)
    delta_log = log_shocks * beta_sum

    for index, coicop in enumerate(data.coicops):
        if coicop in SHOCK_COICOPS:
            delta_log[:, index] = log_shocks[:, 0] * lag0[:, index]

    scenario = baseline * np.exp(delta_log)

    origin_wide = (
        data.history[data.history["date"] == data.origin]
        .pivot(index="country", columns="coicop", values="index_value")
        .reindex(index=data.countries, columns=data.coicops)
        .astype(float)
    )
    origin_index = origin_wide.to_numpy(dtype=float)
    shocked_origin = origin_index.copy()
    for index, coicop in enumerate(data.coicops):
        if coicop in SHOCK_COICOPS:
            shocked_origin[:, index] = (
                origin_index[:, index] * (1.0 + percentages / 100.0)
            )

    rows = []
    for country_index, country in enumerate(data.countries):
        for component_index, coicop in enumerate(data.coicops):
            base_value = float(baseline[country_index, component_index])
            scenario_value = float(scenario[country_index, component_index])
            rows.append(
                {
                    "country": country,
                    "coicop": coicop,
                    "baseline_index": base_value,
                    "scenario_index": scenario_value,
                    "delta_log": float(
                        delta_log[country_index, component_index]
                    ),
                    "scenario_pct": 100.0
                    * float(
                        np.expm1(delta_log[country_index, component_index])
                    ),
                    "origin_index": float(
                        origin_index[country_index, component_index]
                    ),
                    "shocked_origin_index": float(
                        shocked_origin[country_index, component_index]
                    ),
                    "is_source": coicop in SHOCK_COICOPS,
                    "shock_percent": float(percentages[country_index]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_scenarios(
    data: DashboardData,
    scenario: pd.DataFrame,
) -> AggregationResult:
    """Aggregate components to country and EA headline measures."""
    scenario_wide = (
        scenario.pivot(
            index="country", columns="coicop", values="scenario_index"
        )
        .reindex(index=data.countries, columns=data.coicops)
        .astype(float)
    )
    baseline = data.baseline_wide
    item_weights = data.item_norm_wide
    country_weights = data.country_norm

    country_baseline = (item_weights * baseline).sum(axis=1)
    country_scenario = (item_weights * scenario_wide).sum(axis=1)
    country_delta = country_scenario - country_baseline
    country_headlines = pd.DataFrame(
        {
            "country": data.countries,
            "baseline": country_baseline.to_numpy(dtype=float),
            "scenario": country_scenario.to_numpy(dtype=float),
            "delta": country_delta.to_numpy(dtype=float),
            "pct_change": (
                100.0 * (country_scenario / country_baseline - 1.0)
            ).to_numpy(dtype=float),
        }
    ).set_index("country")

    country_component_delta = item_weights * (scenario_wide - baseline)
    country_components = country_component_delta.rename_axis(
        index="country", columns="coicop"
    ).stack().rename("contribution").reset_index()
    country_components = country_components.merge(
        baseline.rename_axis(index="country", columns="coicop")
        .stack()
        .rename("baseline")
        .reset_index(),
        on=["country", "coicop"],
        how="left",
    ).merge(
        scenario_wide.rename_axis(index="country", columns="coicop")
        .stack()
        .rename("scenario")
        .reset_index(),
        on=["country", "coicop"],
        how="left",
    )
    country_components["delta"] = (
        country_components["scenario"] - country_components["baseline"]
    )

    combined_weights = item_weights.mul(country_weights, axis=0)
    ea_item_weight = combined_weights.sum(axis=0)
    ea_baseline_numerator = (combined_weights * baseline).sum(axis=0)
    ea_scenario_numerator = (combined_weights * scenario_wide).sum(axis=0)
    ea_component_baseline = ea_baseline_numerator / ea_item_weight
    ea_component_scenario = ea_scenario_numerator / ea_item_weight
    ea_component_delta = ea_component_scenario - ea_component_baseline
    ea_component_contribution = ea_item_weight * ea_component_delta
    ea_components = pd.DataFrame(
        {
            "coicop": data.coicops,
            "baseline": ea_component_baseline.to_numpy(dtype=float),
            "scenario": ea_component_scenario.to_numpy(dtype=float),
            "delta": ea_component_delta.to_numpy(dtype=float),
            "contribution": ea_component_contribution.to_numpy(dtype=float),
        }
    )

    ea_country_contribution = (
        country_weights * country_delta
    ).rename("contribution")
    ea_countries = pd.DataFrame(
        {
            "country": data.countries,
            "baseline": country_baseline.to_numpy(dtype=float),
            "scenario": country_scenario.to_numpy(dtype=float),
            "delta": country_delta.to_numpy(dtype=float),
            "contribution": ea_country_contribution.to_numpy(dtype=float),
        }
    )

    ea_baseline = float((country_weights * country_baseline).sum())
    ea_scenario = float((country_weights * country_scenario).sum())
    ea_headline = {
        "baseline": ea_baseline,
        "scenario": ea_scenario,
        "delta": ea_scenario - ea_baseline,
        "pct_change": 100.0 * (ea_scenario / ea_baseline - 1.0),
    }

    return AggregationResult(
        country_headlines=country_headlines,
        ea_headline=ea_headline,
        country_components=country_components,
        ea_components=ea_components,
        ea_countries=ea_countries,
    )


def _add_shares(frame: pd.DataFrame, total: float) -> pd.DataFrame:
    result = frame.copy()
    gross = result["contribution"].abs().sum()
    if abs(total) > EPSILON:
        result["share_net"] = result["contribution"] / total
    else:
        result["share_net"] = 0.0
    if gross > EPSILON:
        result["share_gross"] = result["contribution"].abs() / gross
    else:
        result["share_gross"] = 0.0
    return result


def selected_headline(
    aggregation: AggregationResult, geo: str
) -> dict[str, float]:
    if geo == EA_GEO:
        return dict(aggregation.ea_headline)
    row = aggregation.country_headlines.loc[geo]
    return {
        "baseline": float(row["baseline"]),
        "scenario": float(row["scenario"]),
        "delta": float(row["delta"]),
        "pct_change": float(row["pct_change"]),
    }


def selected_component_contributions(
    aggregation: AggregationResult, geo: str
) -> pd.DataFrame:
    if geo == EA_GEO:
        frame = aggregation.ea_components.copy()
    else:
        frame = aggregation.country_components[
            aggregation.country_components["country"] == geo
        ].copy()
    total = selected_headline(aggregation, geo)["delta"]
    return _add_shares(frame, total)


def selected_country_contributions(
    aggregation: AggregationResult, geo: str
) -> pd.DataFrame:
    if geo == EA_GEO:
        frame = aggregation.ea_countries.copy()
        total = aggregation.ea_headline["delta"]
    else:
        row = aggregation.country_headlines.loc[geo]
        frame = pd.DataFrame(
            [
                {
                    "country": geo,
                    "baseline": float(row["baseline"]),
                    "scenario": float(row["scenario"]),
                    "delta": float(row["delta"]),
                    "contribution": float(row["delta"]),
                }
            ]
        )
        total = float(row["delta"])
    return _add_shares(frame, total)


def headline_history(
    data: DashboardData, geo: str
) -> pd.DataFrame:
    frame = data.headline_history[
        data.headline_history["country"] == geo
    ].sort_values("date")
    return frame[["date", "index_value"]].reset_index(drop=True)


def component_history(
    data: DashboardData, geo: str
) -> pd.DataFrame:
    if geo == EA_GEO:
        source = data.ea_component_history
    else:
        source = data.history[data.history["country"] == geo]
    return (
        source.pivot(index="date", columns="coicop", values="index_value")
        .reindex(columns=data.coicops)
        .sort_index()
    )


def component_levels(
    aggregation: AggregationResult, geo: str
) -> pd.DataFrame:
    if geo == EA_GEO:
        frame = aggregation.ea_components.copy()
    else:
        frame = aggregation.country_components[
            aggregation.country_components["country"] == geo
        ].copy()
    return frame.set_index("coicop")[["baseline", "scenario", "delta"]]


def selected_graph_series(
    data: DashboardData,
    aggregation: AggregationResult,
    geo: str,
    series: str,
    shock_percent: float | None = None,
) -> GraphSeries:
    """Return the history and target levels for one graph series."""
    if series == "HEADLINE":
        history = headline_history(data, geo)
        levels = selected_headline(aggregation, geo)
    else:
        if series not in data.coicops:
            raise ValueError(f"Unknown graph series: {series}")
        history = (
            component_history(data, geo)[series]
            .dropna()
            .rename("index_value")
            .reset_index()
        )
        row = component_levels(aggregation, geo).loc[series]
        levels = {
            "baseline": float(row["baseline"]),
            "scenario": float(row["scenario"]),
        }

    origin_rows = history.loc[history["date"] == data.origin, "index_value"]
    if origin_rows.empty:
        raise ValueError(f"No origin observation for {geo}/{series}")
    return GraphSeries(
        code=series,
        history=history[["date", "index_value"]].reset_index(drop=True),
        origin_index=float(origin_rows.iloc[0]),
        baseline=float(levels["baseline"]),
        scenario=float(levels["scenario"]),
        is_source=series in SHOCK_COICOPS,
        shock_percent=shock_percent,
    )


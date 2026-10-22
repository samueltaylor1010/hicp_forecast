from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.scenario_engine import (
    EA_GEO,
    SHOCK_COICOPS,
    aggregate_scenarios,
    headline_history,
    load_dashboard_data,
    percent_to_log_shock,
    scenario_components,
    selected_component_contributions,
    selected_country_contributions,
    selected_graph_series,
    selected_headline,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data"


@pytest.fixture(scope="module")
def data():
    return load_dashboard_data(DATA_DIR)


def _uniform(percent: float, countries: list[str]) -> dict[str, float]:
    return {country: percent for country in countries}


def test_percent_to_log_shock():
    assert percent_to_log_shock(0.0) == pytest.approx(0.0)
    assert percent_to_log_shock(10.0) == pytest.approx(np.log(1.10))
    assert percent_to_log_shock(-10.0) == pytest.approx(np.log(0.90))
    with pytest.raises(ValueError):
        percent_to_log_shock(-100.0)


def test_positive_shock_uses_origin_index_percent(data):
    scenario = scenario_components(data, _uniform(10.0, data.countries))
    fr = scenario[scenario["country"] == "FR"].set_index("coicop")
    log_shock = np.log(1.10)

    assert fr.loc["CP045", "delta_log"] == pytest.approx(
        log_shock * (1.0 + data.lag0_wide.loc["FR", "CP045"])
    )
    assert fr.loc["CP0722", "delta_log"] == pytest.approx(
        log_shock * (1.0 + data.lag0_wide.loc["FR", "CP0722"])
    )
    assert fr.loc["CP01", "delta_log"] == pytest.approx(
        log_shock * data.beta_sum_wide.loc["FR", "CP01"]
    )
    assert fr.loc["CP01", "scenario_index"] == pytest.approx(
        fr.loc["CP01", "baseline_index"]
        * np.exp(fr.loc["CP01", "delta_log"])
    )
    for code in SHOCK_COICOPS:
        assert fr.loc[code, "shocked_origin_index"] == pytest.approx(
            fr.loc[code, "origin_index"] * 1.10
        )


def test_source_shock_includes_origin_level_effect(data):
    scenario = scenario_components(data, _uniform(10.0, data.countries))
    fr = scenario[scenario["country"] == "FR"].set_index("coicop")
    log_shock = np.log(1.10)

    for code in SHOCK_COICOPS:
        expected = log_shock * (1.0 + data.lag0_wide.loc["FR", code])
        assert fr.loc[code, "delta_log"] == pytest.approx(expected)


def test_cli_equivalent_log_shock(data):
    cli_log_shock = 0.10
    equivalent_percent = 100.0 * np.expm1(cli_log_shock)
    scenario = scenario_components(
        data, _uniform(equivalent_percent, data.countries)
    )
    fr = scenario[scenario["country"] == "FR"].set_index("coicop")

    assert fr.loc["CP01", "delta_log"] == pytest.approx(
        cli_log_shock * data.beta_sum_wide.loc["FR", "CP01"]
    )
    assert fr.loc["CP045", "delta_log"] == pytest.approx(
        cli_log_shock * (1.0 + data.lag0_wide.loc["FR", "CP045"])
    )


def test_dashboard_ten_percent_is_cli_log_ten_percent(data):
    scenario = scenario_components(data, _uniform(10.0, data.countries))
    fr = scenario[scenario["country"] == "FR"].set_index("coicop")
    cli_equivalent = np.log1p(0.10)

    assert fr.loc["CP01", "delta_log"] == pytest.approx(
        cli_equivalent * data.beta_sum_wide.loc["FR", "CP01"]
    )


def test_contributions_sum_to_country_and_ea_totals(data):
    scenario = scenario_components(data, _uniform(10.0, data.countries))
    aggregation = aggregate_scenarios(data, scenario)

    fr_total = selected_headline(aggregation, "FR")["delta"]
    fr_components = selected_component_contributions(aggregation, "FR")
    assert fr_components["contribution"].sum() == pytest.approx(
        fr_total, abs=1e-10
    )

    ea_total = aggregation.ea_headline["delta"]
    ea_components = selected_component_contributions(aggregation, EA_GEO)
    ea_countries = selected_country_contributions(aggregation, EA_GEO)
    assert ea_components["contribution"].sum() == pytest.approx(
        ea_total, abs=1e-10
    )
    assert ea_countries["contribution"].sum() == pytest.approx(
        ea_total, abs=1e-10
    )


def test_country_specific_shock_flows_through_weights(data):
    shocks = {country: 0.0 for country in data.countries}
    shocks["DE"] = 10.0
    scenario = scenario_components(data, shocks)
    aggregation = aggregate_scenarios(data, scenario)

    de_delta = selected_headline(aggregation, "DE")["delta"]
    expected_ea_delta = data.country_norm.loc["DE"] * de_delta
    assert aggregation.ea_headline["delta"] == pytest.approx(
        expected_ea_delta, abs=1e-12
    )

    ea_countries = selected_country_contributions(aggregation, EA_GEO)
    nonzero = ea_countries[ea_countries["contribution"].abs() > 1e-12]
    assert list(nonzero["country"]) == ["DE"]


def test_zero_shock_has_no_contributions(data):
    scenario = scenario_components(data, _uniform(0.0, data.countries))
    aggregation = aggregate_scenarios(data, scenario)
    assert aggregation.ea_headline["delta"] == pytest.approx(0.0, abs=1e-12)
    components = selected_component_contributions(aggregation, EA_GEO)
    assert components["contribution"].abs().max() < 1e-12


def test_negative_shock_and_negative_contributions(data):
    scenario = scenario_components(data, _uniform(-10.0, data.countries))
    aggregation = aggregate_scenarios(data, scenario)
    assert aggregation.ea_headline["delta"] < 0
    components = selected_component_contributions(aggregation, EA_GEO)
    assert components["contribution"].sum() == pytest.approx(
        aggregation.ea_headline["delta"], abs=1e-10
    )
    assert (components["share_gross"] >= 0).all()
    assert components["share_gross"].sum() == pytest.approx(1.0)


def test_all_geographies_have_headline_and_component_history(data):
    for geo in data.geos:
        scenario = scenario_components(data, _uniform(10.0, data.countries))
        aggregation = aggregate_scenarios(data, scenario)
        headline = selected_headline(aggregation, geo)
        assert np.isfinite(headline["delta"])
        components = selected_component_contributions(aggregation, geo)
        assert len(components) > 0
        countries = selected_country_contributions(aggregation, geo)
        assert len(countries) > 0


def test_selected_graph_series_matches_aggregation(data):
    scenario = scenario_components(data, _uniform(10.0, data.countries))
    aggregation = aggregate_scenarios(data, scenario)

    headline = selected_graph_series(
        data, aggregation, "FR", "HEADLINE", 10.0
    )
    levels = selected_headline(aggregation, "FR")
    expected_origin = headline_history(data, "FR").loc[
        lambda frame: frame["date"] == data.origin, "index_value"
    ].iloc[0]
    assert headline.origin_index == pytest.approx(expected_origin)
    assert headline.baseline == pytest.approx(levels["baseline"])
    assert headline.scenario == pytest.approx(levels["scenario"])

    component = selected_graph_series(
        data, aggregation, "FR", "CP045", 10.0
    )
    assert component.is_source
    assert component.shock_percent == pytest.approx(10.0)
    assert component.origin_index == pytest.approx(
        data.history[
            (data.history["country"] == "FR")
            & (data.history["date"] == data.origin)
            & (data.history["coicop"] == "CP045")
        ]["index_value"].iloc[0]
    )

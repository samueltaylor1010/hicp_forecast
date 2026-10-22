from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.charts import series_figure
from app.scenario_engine import (
    EA_GEO,
    aggregate_scenarios,
    load_dashboard_data,
    scenario_components,
    selected_graph_series,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data"


def test_app_data_files_exist():
    expected = {
        "baselines.csv",
        "multipliers.csv",
        "historical_indices.csv",
        "weights.csv",
        "headline_history.csv",
        "ea_component_history.csv",
    }
    assert expected <= {path.name for path in DATA_DIR.iterdir()}


def test_weights_are_complete_and_normalised():
    weights = pd.read_csv(DATA_DIR / "weights.csv")
    assert weights["country"].nunique() == 21
    assert weights["coicop"].nunique() == 15
    assert len(weights) == 315
    assert weights["weight_year"].nunique() == 1
    assert not weights.duplicated(["country", "coicop"]).any()
    item_sums = weights.groupby("country")["item_weight"].sum()
    assert np.allclose(item_sums, 1000.0, atol=0.1)
    country_sums = weights.groupby("country")["country_weight"].first().sum()
    assert np.isclose(country_sums, 1000.0, atol=0.1)


def test_official_histories_cover_all_geographies():
    headline = pd.read_csv(DATA_DIR / "headline_history.csv", parse_dates=["date"])
    assert headline["country"].nunique() == 22
    assert headline.groupby("country").size().min() == 36
    assert not headline.duplicated(["country", "date"]).any()

    ea_components = pd.read_csv(
        DATA_DIR / "ea_component_history.csv", parse_dates=["date"]
    )
    assert ea_components["coicop"].nunique() == 15
    assert ea_components.groupby("coicop").size().min() == 36
    assert not ea_components.duplicated(["coicop", "date"]).any()


def test_charts_render_for_all_series_and_selected_geographies():
    data = load_dashboard_data(DATA_DIR)
    scenario = scenario_components(
        data, {country: 10.0 for country in data.countries}
    )
    aggregation = aggregate_scenarios(data, scenario)

    for geo in ["FR", "DE", EA_GEO]:
        for series in ["HEADLINE", *data.coicops]:
            graph = selected_graph_series(
                data,
                aggregation,
                geo,
                series,
                10.0,
            )
            series_figure(data, graph, geo)

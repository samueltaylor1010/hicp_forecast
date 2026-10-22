from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hicp_forecast


def _synthetic_series(periods: int = 108) -> pd.Series:
    index = pd.date_range("2015-01-01", periods=periods, freq="MS")
    values = np.sin(np.arange(periods) / 5.0) * 0.01
    return pd.Series(values, index=index)


def test_training_frame_uses_cumulative_yoy_target_and_monthly_changes():
    index = pd.date_range("2020-01-01", periods=40, freq="MS")
    yoy = pd.Series(np.arange(40, dtype=float) / 100.0, index=index)
    dy = yoy.diff()
    shock = pd.Series(np.sin(np.arange(40) / 3.0), index=index)

    data = hicp_forecast._training_frame(yoy, dy, {"CP045": shock})
    date = index[20]
    row = data.loc[date]

    assert row["target"] == pytest.approx(yoy.loc[index[23]] - yoy.loc[date])
    assert row["dy0"] == pytest.approx(dy.loc[date])
    assert row["dy1"] == pytest.approx(
        dy.loc[date - pd.DateOffset(months=1)]
    )
    assert row["dy2"] == pytest.approx(
        dy.loc[date - pd.DateOffset(months=2)]
    )
    assert row["dy3"] == pytest.approx(
        dy.loc[date - pd.DateOffset(months=3)]
    )
    assert row["dy12"] == pytest.approx(
        dy.loc[date - pd.DateOffset(months=12)]
    )
    assert row["shock_CP045"] == pytest.approx(shock.loc[date])


def test_forecast_features_order_and_values():
    index = pd.date_range("2020-01-01", periods=40, freq="MS")
    dy = pd.Series(np.arange(40, dtype=float) / 100.0, index=index)
    shock = pd.Series(np.arange(40, dtype=float) / 50.0, index=index)
    origin = index[25]

    features = hicp_forecast._forecast_features(
        dy, {"CP045": shock}, origin
    )
    expected = np.array(
        [
            1.0,
            shock.loc[origin],
            *[
                dy.loc[origin - pd.DateOffset(months=offset)]
                for offset in (0, 1, 2, 3, 12)
            ],
        ]
    )

    np.testing.assert_allclose(features, expected)


def test_solver_matches_ols_when_unbounded():
    rng = np.random.default_rng(7)
    rows = 100
    data = pd.DataFrame(
        {
            "target": rng.normal(size=rows),
            "shock_CP045": rng.normal(size=rows),
            "shock_CP0722": rng.normal(size=rows),
            "dy0": rng.normal(size=rows),
            "dy1": rng.normal(size=rows),
            "dy2": rng.normal(size=rows),
            "dy3": rng.normal(size=rows),
            "dy12": rng.normal(size=rows),
        }
    )
    columns = [
        "shock_CP045",
        "shock_CP0722",
        "dy0",
        "dy1",
        "dy2",
        "dy3",
        "dy12",
    ]
    design = np.column_stack(
        [np.ones(rows), data[columns].to_numpy(dtype=float)]
    )
    target = data["target"].to_numpy(dtype=float)
    expected, *_ = np.linalg.lstsq(design, target, rcond=None)

    coefficients = hicp_forecast._solve_model(
        data, ("CP045", "CP0722")
    )

    np.testing.assert_allclose(coefficients, expected, atol=1e-9)


def test_fit_reconstructs_target_yoy_before_index_inversion(monkeypatch):
    index = pd.date_range("2015-01-01", periods=108, freq="MS")
    trend = np.arange(len(index), dtype=float)
    wide = pd.DataFrame(
        {
            ("FR", "CP01"): 100.0
            * np.exp(0.001 * trend + 0.01 * np.sin(trend / 6.0)),
            ("FR", "CP045"): 100.0
            * np.exp(0.002 * trend + 0.02 * np.sin(trend / 5.0)),
            ("FR", "CP0722"): 100.0
            * np.exp(0.0015 * trend + 0.015 * np.cos(trend / 7.0)),
        },
        index=index,
    )
    wide.columns = pd.MultiIndex.from_tuples(
        wide.columns, names=["geo", "coicop"]
    )
    yoy = np.log(wide / wide.shift(12))
    dy = yoy.diff()
    origin = index[80]
    coefficients = np.array(
        [0.001, 0.2, -0.1, 0.3, 0.1, 0.05, 0.02, -0.01]
    )

    monkeypatch.setattr(
        hicp_forecast,
        "_solve_model",
        lambda data, shock_names: coefficients.copy(),
    )

    baseline, multipliers = hicp_forecast.fit_direct_model(
        wide,
        yoy,
        dy,
        "FR",
        "CP01",
        origin,
        ("CP045", "CP0722"),
    )

    feature_row = np.array(
        [
            1.0,
            dy.loc[origin, ("FR", "CP045")],
            dy.loc[origin, ("FR", "CP0722")],
            *[
                dy.loc[
                    origin - pd.DateOffset(months=offset), ("FR", "CP01")
                ]
                for offset in (0, 1, 2, 3, 12)
            ],
        ]
    )
    delta_yoy = float(feature_row @ coefficients)
    expected_yoy = float(yoy.loc[origin, ("FR", "CP01")]) + delta_yoy
    target_period = origin + pd.DateOffset(months=3)
    base_index = float(
        wide.loc[target_period - pd.DateOffset(months=12), ("FR", "CP01")]
    )

    assert baseline["baseline_index_t3"] == pytest.approx(
        base_index * np.exp(expected_yoy)
    )
    assert baseline["lag0_coefficient"] == pytest.approx(coefficients[3])
    assert [row["beta_multiplier"] for row in multipliers] == pytest.approx(
        [coefficients[1], coefficients[2]]
    )

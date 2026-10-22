# HICP energy shock dashboard.

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd
from shiny import reactive
from shiny.express import input, render, ui

try:
    from .charts import series_figure
    from .labels import (
        component_color_map,
        component_label,
        country_color_map,
        country_label,
    )
    from .scenario_engine import (
        EA_GEO,
        GraphSeries,
        aggregate_scenarios,
        load_dashboard_data,
        scenario_components,
        selected_component_contributions,
        selected_country_contributions,
        selected_graph_series,
        selected_headline,
    )
except ImportError:  # Shinylive loads the app directory as top-level modules.
    from charts import series_figure
    from labels import (
        component_color_map,
        component_label,
        country_color_map,
        country_label,
    )
    from scenario_engine import (
        EA_GEO,
        GraphSeries,
        aggregate_scenarios,
        load_dashboard_data,
        scenario_components,
        selected_component_contributions,
        selected_country_contributions,
        selected_graph_series,
        selected_headline,
    )

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA = load_dashboard_data(DATA_DIR)

COUNTRY_CHOICES = {
    code: f"{country_label(code)} ({code})" for code in DATA.countries
}
GEO_CHOICES = {
    EA_GEO: "Euro area (EA)",
    **COUNTRY_CHOICES,
}
SERIES_CHOICES = {
    "HEADLINE": "Headline HICP",
    **{
        code: f"{code} — {component_label(code)}"
        for code in DATA.coicops
    },
}
COMPONENT_COLORS = component_color_map(list(DATA.coicops))
COUNTRY_COLORS = country_color_map(list(DATA.countries))
DEFAULT_SHOCK = 10.0


def _fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _parse_shock_text(raw: str | None) -> tuple[float | None, str | None]:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None, "Enter a number."
    if not math.isfinite(value):
        return None, "Enter a finite number."
    if value <= -100.0:
        return None, "Shock must be greater than -100%."
    return value, None


def _fmt_input(value: float) -> str:
    return f"{value:g}"


def _metrics_html(graph: GraphSeries) -> str:
    baseline_change = graph.baseline - graph.origin_index
    scenario_change = graph.scenario - graph.origin_index
    delta = graph.scenario - graph.baseline
    baseline_pct = 100.0 * (graph.baseline / graph.origin_index - 1.0)
    scenario_pct = 100.0 * (graph.scenario / graph.origin_index - 1.0)
    delta_pct = 100.0 * (graph.scenario / graph.baseline - 1.0)
    return f"""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Baseline change</div>
        <div class="metric-value baseline">{_fmt_signed(baseline_change)} pts</div>
        <div class="metric-sub">
          {_fmt_signed(baseline_pct)}% vs origin {DATA.origin:%Y-%m}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Shocked change</div>
        <div class="metric-value shocked">{_fmt_signed(scenario_change)} pts</div>
        <div class="metric-sub">
          {_fmt_signed(scenario_pct)}% vs origin {DATA.origin:%Y-%m}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Delta</div>
        <div class="metric-value delta">{_fmt_signed(delta)} pts</div>
        <div class="metric-sub">
          {_fmt_signed(delta_pct)}% shocked vs baseline
        </div>
      </div>
    </div>
    """


def _segment_html(
    *,
    label: str,
    value: float,
    share: float,
    share_net: float | None,
    color: str,
    bottom: float,
    height: float,
) -> str:
    net_text = (
        f", {share_net * 100:+.1f}% of net change"
        if share_net is not None
        else ""
    )
    tooltip = f"{label}: {_fmt_signed(value)} index pts ({share:.1f}% of gross){net_text}"
    escaped_tooltip = html.escape(tooltip)
    return (
        f'<div class="bar-segment" tabindex="0" '
        f'style="bottom:{bottom:.4f}%;height:{height:.4f}%;'
        f'background-color:{color};" '
        f'data-tooltip="{escaped_tooltip}" title="{escaped_tooltip}"></div>'
    )


def _bar_html(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label_fn,
    color_map: dict[str, str],
    title: str,
    total: float,
    total_pct: float,
    card_class: str,
) -> str:
    clean = frame.loc[frame["contribution"].abs() > 1e-12].copy()
    if clean.empty:
        return f"""
        <div class="contrib-card {html.escape(card_class)}">
          <div class="contrib-head">
            <span>{html.escape(title)}</span>
            <span class="total-value">0.00 pts</span>
          </div>
          <div class="empty-bar">No change above baseline.</div>
        </div>
        """

    clean = clean.sort_values("contribution", ascending=False)
    positive = clean[clean["contribution"] > 0].copy()
    negative = clean[clean["contribution"] < 0].copy()

    gross = float(clean["contribution"].abs().sum())
    negative_abs_sum = float(negative["contribution"].abs().sum())
    zero_position = 100.0 * negative_abs_sum / gross if gross else 0.0

    segments = []
    cumulative = 0.0
    for _, row in positive.iterrows():
        code = str(row[label_column])
        height = 100.0 * float(row["contribution"]) / gross
        segments.append(
            _segment_html(
                label=label_fn(code),
                value=float(row["contribution"]),
                share=100.0 * abs(float(row["contribution"])) / gross,
                share_net=(
                    float(row["share_net"])
                    if "share_net" in row.index
                    else None
                ),
                color=color_map.get(code, "#95a5a6"),
                bottom=zero_position + cumulative,
                height=height,
            )
        )
        cumulative += height

    cumulative = 0.0
    for _, row in negative.sort_values("contribution").iterrows():
        code = str(row[label_column])
        height = 100.0 * abs(float(row["contribution"])) / gross
        segments.append(
            _segment_html(
                label=label_fn(code),
                value=float(row["contribution"]),
                share=100.0 * abs(float(row["contribution"])) / gross,
                share_net=(
                    float(row["share_net"])
                    if "share_net" in row.index
                    else None
                ),
                color=color_map.get(code, "#95a5a6"),
                bottom=zero_position - cumulative - height,
                height=height,
            )
        )
        cumulative += height

    legend = []
    for _, row in clean.sort_values(
        "contribution", key=lambda values: values.abs(), ascending=False
    ).iterrows():
        code = str(row[label_column])
        share_net = (
            float(row["share_net"]) if "share_net" in row.index else None
        )
        net_text = (
            f"{share_net * 100:+.1f}% net"
            if share_net is not None
            else ""
        )
        legend.append(
            f"""
            <div class="legend-row" title="{html.escape(label_fn(code))}">
              <span class="swatch" style="background:{color_map.get(code, "#95a5a6")};"></span>
              <span class="legend-label">{html.escape(label_fn(code))}</span>
              <span class="legend-value {("pos" if row["contribution"] >= 0 else "neg")}">
                {_fmt_signed(float(row["contribution"]))}
              </span>
              <span class="legend-share">{net_text}</span>
            </div>
            """
        )

    zero_line = ""
    if 0.0 < zero_position < 100.0:
        zero_line = (
            f'<div class="zero-line" style="bottom:{zero_position:.4f}%;"></div>'
        )

    return f"""
    <div class="contrib-card {html.escape(card_class)}">
      <div class="contrib-head">
        <span>{html.escape(title)}</span>
        <span class="total-value">{_fmt_signed(total)} pts
          <span class="total-sub">({_fmt_signed(total_pct)}%)</span>
        </span>
      </div>
      <div class="bar-area">
        <div class="stack-bar">
          {''.join(segments)}
          {zero_line}
        </div>
      </div>
      <div class="legend-scroll">
        {''.join(legend)}
      </div>
    </div>
    """


uniform_shock = reactive.value(DEFAULT_SHOCK)
shock_values = reactive.value(
    {country: 0.0 for country in DATA.countries}
)


@reactive.calc
def effective_shocks() -> dict[str, float]:
    if input.scope() == "uniform":
        value = float(uniform_shock())
        return {country: value for country in DATA.countries}
    return dict(shock_values())


@reactive.calc
def scenario() -> pd.DataFrame:
    return scenario_components(DATA, effective_shocks())


@reactive.calc
def aggregation():
    return aggregate_scenarios(DATA, scenario())


@reactive.calc
def graph_series():
    geo = input.geo()
    shocks = effective_shocks()
    shock_percent = (
        float(shocks[geo]) if geo in DATA.countries else None
    )
    return selected_graph_series(
        DATA,
        aggregation(),
        geo,
        input.series(),
        shock_percent,
    )


@reactive.effect
@reactive.event(input.shock)
def _store_shock_input() -> None:
    value, error = _parse_shock_text(input.shock())
    if error is not None or value is None:
        return
    if input.scope() == "uniform":
        uniform_shock.set(value)
        return
    state = dict(shock_values())
    state[input.shock_country()] = value
    shock_values.set(state)


@reactive.effect
@reactive.event(input.geo)
def _sync_shock_country_with_geo() -> None:
    if input.scope() != "country":
        return
    geo = input.geo()
    if geo not in DATA.countries:
        return
    ui.update_select("shock_country", selected=geo)
    ui.update_text(
        "shock",
        value=_fmt_input(float(shock_values().get(geo, 0.0))),
    )


@reactive.effect
@reactive.event(input.shock_country)
def _sync_text_to_country() -> None:
    if input.scope() != "country":
        return
    country = input.shock_country()
    ui.update_text(
        "shock",
        value=_fmt_input(float(shock_values().get(country, 0.0))),
    )


@reactive.effect
@reactive.event(input.scope)
def _sync_text_to_scope() -> None:
    if input.scope() == "uniform":
        ui.update_text("shock", value=_fmt_input(float(uniform_shock())))
    else:
        country = input.shock_country()
        ui.update_text(
            "shock",
            value=_fmt_input(float(shock_values().get(country, 0.0))),
        )


@reactive.effect
@reactive.event(input.apply_all)
def _apply_to_all() -> None:
    value, error = _parse_shock_text(input.shock())
    if error is not None or value is None:
        return
    uniform_shock.set(value)
    shock_values.set({country: value for country in DATA.countries})


@reactive.effect
@reactive.event(input.reset_all)
def _reset_all() -> None:
    uniform_shock.set(DEFAULT_SHOCK)
    shock_values.set(
        {country: DEFAULT_SHOCK for country in DATA.countries}
    )
    ui.update_text("shock", value=_fmt_input(DEFAULT_SHOCK))


@reactive.effect
@reactive.event(input.zero_all)
def _zero_all() -> None:
    uniform_shock.set(0.0)
    shock_values.set({country: 0.0 for country in DATA.countries})
    ui.update_text("shock", value="0")


ui.page_opts(
    fillable=True,
    window_title="HICP shock simulator",
)
ui.include_css(Path(__file__).resolve().parent / "www" / "styles.css")

with ui.div(class_="app-shell"):
    with ui.div(class_="panel panel-left"):
        ui.h1("HICP shock simulator", class_="app-title")
        ui.div(
            "Modelling energy shock effects at a 3 month horizon",
            class_="app-subtitle",
        )

        ui.h2("Controls", class_="section-title")
        ui.input_select(
            "geo",
            "Graph geography",
            choices=GEO_CHOICES,
            selected=EA_GEO,
        )
        ui.input_radio_buttons(
            "scope",
            "Shock scope",
            choices={
                "uniform": "Uniform across EA",
                "country": "By country",
            },
            selected="uniform",
        )
        with ui.panel_conditional("input.scope === 'country'"):
            ui.input_select(
                "shock_country",
                "Shock country",
                choices=COUNTRY_CHOICES,
                selected="FR",
            )
            ui.div(
                "Country-specific mode starts from zero. Set a shock for "
                "the selected country, then switch countries as needed.",
                class_="muted small-note",
            )

        ui.input_text(
            "shock",
            "Shock to latest CP045 + CP0722 index (%)",
            value=_fmt_input(DEFAULT_SHOCK),
            placeholder="e.g. 10",
        )

        @render.ui
        def shock_error():
            _, error = _parse_shock_text(input.shock())
            if error is None:
                return ui.HTML("")
            return ui.div(error, class_="input-error")

        with ui.div(class_="button-row"):
            ui.input_action_button(
                "apply_all",
                "Apply current value to all",
                class_="btn-secondary btn-sm",
            )
            ui.input_action_button(
                "zero_all",
                "Zero all shocks",
                class_="btn-secondary btn-sm",
            )
            ui.input_action_button(
                "reset_all",
                "Reset all to +10%",
                class_="btn-secondary btn-sm",
            )

    with ui.div(class_="panel panel-center"):
        with ui.div(class_="center-head"):
            ui.h2("Scenario graph", class_="section-title")
            ui.input_select(
                "series",
                "Series",
                choices=SERIES_CHOICES,
                selected="HEADLINE",
            )

        @render.ui
        def graph_metrics():
            return ui.HTML(_metrics_html(graph_series()))

        @render.plot(height=620)
        def graph_plot():
            return series_figure(DATA, graph_series(), input.geo())

        ui.div(
            (
                "Scenario indices apply a percent shock to the latest "
                "CP045/CP0722 index, then propagate it through the "
                "direct-horizon coefficients. EA aggregates are built "
                "bottom-up with country and item weights."
            ),
            class_="method-note",
        )

    with ui.div(class_="panel panel-right"):
        ui.h2("Contribution to the increase", class_="section-title")

        with ui.div(class_="bars-grid"):
            @render.ui
            def component_bar():
                geo = input.geo()
                frame = selected_component_contributions(aggregation(), geo)
                total = selected_headline(aggregation(), geo)
                return ui.HTML(
                    _bar_html(
                        frame,
                        label_column="coicop",
                        label_fn=lambda code: (
                            f"{code} — {component_label(code)}"
                            if component_label(code) != code
                            else code
                        ),
                        color_map=COMPONENT_COLORS,
                        title="By component",
                        total=total["delta"],
                        total_pct=total["pct_change"],
                        card_class="component-card",
                    )
                )

            @render.ui
            def country_bar():
                geo = input.geo()
                frame = selected_country_contributions(aggregation(), geo)
                total = selected_headline(aggregation(), geo)
                return ui.HTML(
                    _bar_html(
                        frame,
                        label_column="country",
                        label_fn=lambda code: (
                            f"{country_label(code)} ({code})"
                            if country_label(code) != code
                            else code
                        ),
                        color_map=COUNTRY_COLORS,
                        title="By country",
                        total=total["delta"],
                        total_pct=total["pct_change"],
                        card_class="country-card",
                    )
                )


        ui.div(
            (
                "Bars show index-point contributions. Segment height is "
                "proportional to absolute contribution; negative "
                "contributions sit below the zero line."
            ),
            class_="method-note",
        )

"""Interactive HICP energy shock simulation using country and item weights."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from shiny import reactive
from shiny.express import input, render, ui

try:
    from .charts import component_grid_figure, headline_figure
    from .labels import (
        component_color_map,
        component_label,
        country_color_map,
        country_label,
    )
    from .scenario_engine import (
        EA_GEO,
        aggregate_scenarios,
        load_dashboard_data,
        scenario_components,
        selected_component_contributions,
        selected_country_contributions,
        selected_headline,
        source_origin_indices,
    )
except ImportError:  # Shinylive loads the app directory as top-level modules.
    from charts import component_grid_figure, headline_figure
    from labels import (
        component_color_map,
        component_label,
        country_color_map,
        country_label,
    )
    from scenario_engine import (
        EA_GEO,
        aggregate_scenarios,
        load_dashboard_data,
        scenario_components,
        selected_component_contributions,
        selected_country_contributions,
        selected_headline,
        source_origin_indices,
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
COMPONENT_COLORS = component_color_map(list(DATA.coicops))
COUNTRY_COLORS = country_color_map(list(DATA.countries))
DEFAULT_SHOCK = 10.0


def _fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _kpi_html(geo: str, aggregation) -> str:
    selected = selected_headline(aggregation, geo)
    ea = aggregation.ea_headline
    return f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">{html.escape(country_label(geo))}</div>
        <div class="kpi-value">{_fmt_signed(selected["delta"])} pts</div>
        <div class="kpi-sub">{_fmt_signed(selected["pct_change"])}% vs baseline</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">EA headline</div>
        <div class="kpi-value">{_fmt_signed(ea["delta"])} pts</div>
        <div class="kpi-sub">{_fmt_signed(ea["pct_change"])}% vs baseline</div>
      </div>
    </div>
    """


def _origin_html(geo: str, shock_percent: float) -> str:
    frame = source_origin_indices(DATA, geo, shock_percent)
    note = (
        f"Shock applied to the latest index at "
        f"{DATA.origin:%Y-%m}."
    )
    rows = []
    for _, row in frame.iterrows():
        code = str(row["coicop"])
        rows.append(
            f"""
            <tr>
              <td>{html.escape(code)}</td>
              <td>{row["origin_index"]:.2f}</td>
              <td class="shocked-value">{row["shocked_origin_index"]:.2f}</td>
            </tr>
            """
        )
    return f"""
    <div class="origin-note">
      <div class="origin-note-title">{html.escape(country_label(geo))}</div>
      <div class="muted">{html.escape(note)}</div>
      <table class="origin-table">
        <thead><tr><th>Source</th><th>Origin</th><th>Shocked</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
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
) -> str:
    clean = frame.loc[frame["contribution"].abs() > 1e-12].copy()
    if clean.empty:
        return f"""
        <div class="contrib-card">
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
    <div class="contrib-card">
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


shock_values = reactive.value(
    {country: 0.0 for country in DATA.countries}
)


@reactive.calc
def effective_shocks() -> dict[str, float]:
    if input.scope() == "uniform":
        value = float(input.shock())
        return {country: value for country in DATA.countries}
    return dict(shock_values())


@reactive.calc
def scenario() -> pd.DataFrame:
    return scenario_components(DATA, effective_shocks())


@reactive.calc
def aggregation():
    return aggregate_scenarios(DATA, scenario())


@reactive.effect
@reactive.event(input.geo)
def _sync_shock_country_with_geo() -> None:
    if input.scope() != "country":
        return
    geo = input.geo()
    if geo not in DATA.countries:
        return
    ui.update_select("shock_country", selected=geo)
    ui.update_slider(
        "shock", value=float(shock_values().get(geo, DEFAULT_SHOCK))
    )


@reactive.effect
@reactive.event(input.shock_country)
def _sync_slider_to_country() -> None:
    if input.scope() != "country":
        return
    country = input.shock_country()
    ui.update_slider(
        "shock", value=float(shock_values().get(country, DEFAULT_SHOCK))
    )


@reactive.effect
@reactive.event(input.scope)
def _sync_slider_to_scope() -> None:
    if input.scope() == "country":
        country = input.shock_country()
        ui.update_slider(
            "shock", value=float(shock_values().get(country, DEFAULT_SHOCK))
        )


@reactive.effect
@reactive.event(input.shock)
def _store_country_shock() -> None:
    if input.scope() != "country":
        return
    state = dict(shock_values())
    state[input.shock_country()] = float(input.shock())
    shock_values.set(state)


@reactive.effect
@reactive.event(input.apply_all)
def _apply_to_all() -> None:
    value = float(input.shock())
    shock_values.set({country: value for country in DATA.countries})


@reactive.effect
@reactive.event(input.reset_all)
def _reset_all() -> None:
    shock_values.set(
        {country: DEFAULT_SHOCK for country in DATA.countries}
    )
    ui.update_slider("shock", value=DEFAULT_SHOCK)


@reactive.effect
@reactive.event(input.zero_all)
def _zero_all() -> None:
    shock_values.set({country: 0.0 for country in DATA.countries})
    ui.update_slider("shock", value=0.0)


ui.page_opts(
    title="HICP energy shock dashboard",
    fillable=True,
)
ui.include_css(Path(__file__).resolve().parent / "www" / "styles.css")

with ui.div(class_="app-shell"):
    with ui.div(class_="panel panel-left"):
        ui.h1("HICP shock simulator", class_="app-title")
        ui.div(
            (
                f"Origin {DATA.origin:%Y-%m} → target "
                f"{DATA.target_period:%Y-%m} · weights {DATA.weight_year}"
            ),
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

        ui.h3("Shock to latest CP045 + CP0722 index", class_="control-title")
        ui.div(
            "Percent change applied at the origin index. "
            "The fitted lag/pass-through coefficients then determine the "
            "target-period response.",
            class_="muted small-note",
        )
        with ui.div(class_="vertical-slider-outer"):
            with ui.div(class_="vertical-slider-inner"):
                ui.input_slider(
                    "shock",
                    None,
                    min=-50,
                    max=50,
                    value=DEFAULT_SHOCK,
                    step=1,
                    post="%",
                )

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

        @render.ui
        def origin_summary():
            if input.scope() == "country":
                geo = input.shock_country()
                shock = float(shock_values().get(geo, DEFAULT_SHOCK))
            else:
                geo = input.geo()
                shock = float(input.shock())
            return ui.HTML(_origin_html(geo, shock))

        @render.ui
        def kpis():
            return ui.HTML(_kpi_html(input.geo(), aggregation()))

        ui.div(
            (
                "The country bar is a single segment when a country is "
                "selected. Select Euro area to see the country decomposition."
            ),
            class_="muted footer-note",
        )

    with ui.div(class_="panel panel-center"):
        with ui.div(class_="center-head"):
            ui.h2("Scenario graph", class_="section-title")
            ui.input_radio_buttons(
                "view",
                None,
                choices={
                    "headline": "Headline index",
                    "components": "All components",
                },
                selected="headline",
                inline=True,
            )

        with ui.panel_conditional("input.view === 'headline'"):
            @render.plot(height=520)
            def headline_plot():
                return headline_figure(DATA, aggregation(), input.geo())

        with ui.panel_conditional("input.view === 'components'"):
            @render.plot(height=1080)
            def component_plot():
                shocks = effective_shocks()
                shock_percent = (
                    float(input.shock())
                    if input.scope() == "uniform"
                    else (
                        float(shocks[input.geo()])
                        if input.geo() in DATA.countries
                        else None
                    )
                )
                return component_grid_figure(
                    DATA,
                    aggregation(),
                    input.geo(),
                    shock_percent,
                )

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

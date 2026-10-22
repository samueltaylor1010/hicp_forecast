# hicp_forecast

Forecast HICP components, estimate CP045/CP0722 energy pass-through, and run
an interactive energy-shock simulation dashboard.

## Dashboard

The dashboard is a Shiny Express app in `app/`. It has three panels:

- left: graph geography, uniform vs country-specific shock scope, and a vertical
  percent-shock slider;
- centre: either the selected geography's headline index history plus baseline
  and scenario points, or the full 15-component grid;
- right: index-point contributions to the selected geography's increase, split
  by component and by country.

The shock is a **percent change to the latest origin index of CP045 and CP0722**.
The log shock is propagated to the target period with the fitted direct-horizon
coefficients:

```text
s = log(1 + percent / 100)

pass-through targets: d = s * (beta_CP045 + beta_CP0722)
CP045 / CP0722:        d = s * lag0

scenario index: S = baseline_index * exp(d)
```

EA aggregates are built bottom-up using 2026 country and item weights from
`hicp.db`. The dashboard never reads SQLite at runtime.

### Build dashboard data

```bash
python scripts/build_app_data.py
```

This copies the forecast artifacts and extracts the additional CSV files needed
by the app into `app/data/`:

```text
baselines.csv
multipliers.csv
historical_indices.csv
weights.csv
headline_history.csv
ea_component_history.csv
```

### Run locally

```bash
pip install -r requirements-dev.txt
make app-data
make run
```

Then open the local URL printed by Shiny.

### Export as a static site

```bash
make export
make serve
```

`make export` writes a self-contained Shinylive site to `docs/` and adds
`docs/.nojekyll` for GitHub Pages. The `app/data/` files are bundled into the
static site, so the exported dashboard runs entirely in the browser.

To deploy to GitHub Pages, serve the `docs/` directory from the repository's
Pages settings (or copy it to the Pages branch).

### Tests

```bash
make test
```

The tests cover the scenario math, percent-to-log conversion, EA aggregation
identities, country-specific shocks, negative shocks, all geographies, chart
rendering, and the dashboard data bundle.

The CLI `plot_scenario.py` is not part of the dashboard deliverable. It remains
useful as a methodology oracle. Its `--shock-size` is a log-point shock on the
origin YoY/index, so compare it to the dashboard at:

```text
dashboard percent p  <=>  CLI shock log(1 + p/100)
CLI shock q          <=>  dashboard percent 100 * (exp(q) - 1)
```

For example, the CLI's `0.10` is equivalent to a dashboard shock of `+10.517%`.

## Forecast pipeline

```bash
python hicp_etl.py
python hicp_forecast.py
```

The pipeline writes `baselines.csv`, `multipliers.csv`, and
`historical_indices.csv` in the repository root.

# hicp_forecast

Forecast HICP components, estimate CP045/CP0722 energy pass-through, and run
an interactive energy-shock simulation dashboard.

## Dashboard

Hosted live [here](https://samueltaylor1010.github.io/hicp_forecast/).

The dashboard is a Shiny Express app in `app/`. It has three panels:

- left: graph geography, uniform vs country-specific shock scope, and a
  percent-shock text input;
- centre: a dropdown for the headline index or any single component, three
  origin-to-baseline/scenario metrics, and a single dark-themed index graph;
- right: index-point contributions to the selected geography's increase, split
  by component and by country.

The shock is a **percent change to the latest origin index of CP045 and CP0722**.
The resulting log shock raises both the origin YoY rate and its 1-month change
by `s`. It is propagated to the target period with the fitted direct-horizon
coefficients:

```text
s = log(1 + percent / 100)

target components: d = s * (beta_CP045 + beta_CP0722)
CP045 / CP0722:    d = s * (1 + lag0)

scenario index:    S = baseline_index * exp(d)
```

The extra unit for CP045/CP0722 is the direct shift in the origin YoY level,
which enters because the model forecasts `y_{t+3} - y_t` and the app
reconstructs `y_{t+3} = y_t + delta` before converting back to an index.

### Forecast model

The target is the 3-month cumulative change in the YoY log difference:

```text
y_t  = log(P_t) - log(P_{t-12})
dy_t = y_t - y_{t-1}

y_{t+3} - y_t = alpha
              + beta_1 * ds1_t + beta_2 * ds2_t
              + gamma_0 * dy_t + gamma_1 * dy_{t-1}
              + gamma_2 * dy_{t-2} + gamma_3 * dy_{t-3}
              + gamma_12 * dy_{t-12} + error
```

`ds1_t` and `ds2_t` are the contemporaneous 1-month changes in the CP045 and
CP0722 YoY rates. `lag0_coefficient` in the artifacts is the fitted
`gamma_0` coefficient on `dy_t`. All coefficients are estimated jointly with
`scipy.optimize.lsq_linear` and are unbounded.

EA aggregates are built bottom-up using 2026 country and item weights from
`data/hicp.db`. The dashboard never reads SQLite at runtime.

### Data layout

The source database and model artifacts live in `data/`:

```text
data/
├── hicp.db
├── baselines.csv
├── multipliers.csv
└── historical_indices.csv
```

The generated dashboard bundle is written to `app/data/`. It stays under the
app directory so that Shinylive can bundle the CSVs into the static site.

### Build dashboard data

```bash
python scripts/build_app_data.py
```

This reads the forecast artifacts from `data/` and extracts the additional CSV
files needed by the app into `app/data/`:

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
make pipeline
make run
```

`make pipeline` runs the full ETL, forecast, and dashboard-data pipeline. Use
`make app-data` only when you want to rebuild `app/data/` from the existing
artifacts in `data/`.

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

## Forecast pipeline

```bash
make pipeline
```

or, equivalently:

```bash
python scripts/run_pipeline.py
```

This runs `hicp_etl.py`, then `hicp_forecast.py`, then
`scripts/build_app_data.py`. The ETL writes `data/hicp.db`, the forecast step
writes `baselines.csv`, `multipliers.csv`, and `historical_indices.csv` to
`data/`, and the final step rebuilds `app/data/`.

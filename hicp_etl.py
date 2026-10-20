"""ETL pipeline for Eurostat HICP data into SQLite.

The loaded panel is restricted to the euro area and its current member
states, the headline series, the 13 COICOP divisions, the CP045 and
CP0722 energy sub-components, and the derived CP04_CORE and CP07_CORE
components.
"""

import gzip
import io
import logging
import sqlite3
from datetime import date

import pandas as pd
import requests

API_ROOT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"
ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"
DB_PATH = "hicp.db"
START_YEAR = 1996
COICOP_CODES = (
    "TOTAL",
    "CP01",
    "CP02",
    "CP03",
    "CP04",
    "CP05",
    "CP06",
    "CP07",
    "CP08",
    "CP09",
    "CP10",
    "CP11",
    "CP12",
    "CP13",
    "CP045",
    "CP0722",
)
COICOP_FILTER = ",".join(COICOP_CODES)
GEO_CODES = (
    "EA",
    "AT",
    "BE",
    "BG",
    "CY",
    "DE",
    "EE",
    "ES",
    "FI",
    "FR",
    "EL",
    "HR",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PT",
    "SI",
    "SK",
)
GEO_FILTER = ",".join(GEO_CODES)
CORE_COMPONENTS = {
    "CP04_CORE": ("CP04", "CP045"),
    "CP07_CORE": ("CP07", "CP0722"),
}

SCHEMA = """
CREATE TABLE hicp_panel_data (
    time_period DATE NOT NULL,
    geo VARCHAR(10) NOT NULL,
    coicop VARCHAR(20) NOT NULL,
    hicp_index FLOAT,
    item_weight FLOAT,
    country_weight FLOAT,
    PRIMARY KEY (time_period, geo, coicop)
)
"""


def fetch_csv(
    session: requests.Session,
    dataset: str,
    key: str,
    start: str,
    end: str,
    coicop_filter: str | None = None,
    geo_filter: str | None = None,
) -> pd.DataFrame:
    """Fetch one SDMX CSV time slice as a DataFrame.

    Args:
        session: Requests session carrying the required Accept header.
        dataset: Eurostat dataflow identifier (for example ``PRC_HICP_MINR``).
        key: Dot-separated SDMX series key (for example ``M.I25.*.*``).
        start: Inclusive lower period bound, ``YYYY`` or ``YYYY-MM``.
        end: Inclusive upper period bound, ``YYYY`` or ``YYYY-MM``.
        coicop_filter: Optional comma-separated COICOP codes to request.
        geo_filter: Optional comma-separated geography codes to request.

    Returns:
        Parsed CSV payload.

    Raises:
        RuntimeError: If Eurostat returns a non-CSV payload.
        requests.HTTPError: If the request fails.
    """
    params = {"c[TIME_PERIOD]": f"ge:{start}+le:{end}"}
    if coicop_filter:
        params["c[COICOP18]"] = coicop_filter
    if geo_filter:
        params["c[GEO]"] = geo_filter
    response = session.get(
        f"{API_ROOT}/{dataset}/1.0/{key}",
        params=params,
        timeout=300,
    )
    response.raise_for_status()
    if "csv" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"Unexpected Eurostat response for {dataset}: {response.text[:200]}"
        )
    content = response.content
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return pd.read_csv(
        io.BytesIO(content),
        encoding="utf-8-sig",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

# TODO break some of the workings into sub-function(s) for readability maybe?
def load_year(
    conn: sqlite3.Connection,
    session: requests.Session,
    year: int,
) -> int:
    """Extract, transform, and load one year of HICP observations.

    Args:
        conn: SQLite connection with the target table already created.
        session: Requests session carrying the required Accept header.
        year: Calendar year to process.

    Returns:
        Number of rows appended to ``hicp_panel_data``.
    """
    monthly = fetch_csv(
        session,
        "PRC_HICP_MINR",
        "M.I25.*.*",
        f"{year}-01",
        f"{year}-12",
        coicop_filter=COICOP_FILTER,
        geo_filter=GEO_FILTER,
    )
    items = fetch_csv(
        session,
        "PRC_HICP_IW",
        "A.*.IW.*",
        str(year),
        str(year),
        coicop_filter=COICOP_FILTER,
        geo_filter=GEO_FILTER,
    )
    countries = fetch_csv(
        session,
        "PRC_HICP_CW",
        "A.COWEA.*",
        str(year),
        str(year),
        geo_filter=GEO_FILTER,
    )

    for name, frame in (("monthly", monthly), ("items", items)):
        unexpected = set(frame["coicop18"]) - set(COICOP_CODES)
        if unexpected:
            raise RuntimeError(
                f"Unexpected COICOP codes in {name}: {sorted(unexpected)}"
            )

    for name, frame in (
        ("monthly", monthly),
        ("items", items),
        ("countries", countries),
    ):
        unexpected = set(frame["geo"]) - set(GEO_CODES)
        if unexpected:
            raise RuntimeError(
                f"Unexpected geography codes in {name}: {sorted(unexpected)}"
            )

    monthly["year"] = monthly["TIME_PERIOD"].str[:4].astype(int)
    monthly["time_period"] = monthly["TIME_PERIOD"] + "-01"
    monthly = monthly.rename(
        columns={"coicop18": "coicop", "OBS_VALUE": "hicp_index"}
    )

    items["year"] = year
    items = items.rename(columns={"coicop18": "coicop", "OBS_VALUE": "item_weight"})

    countries["year"] = year
    countries = countries.rename(columns={"OBS_VALUE": "country_weight"})

    panel = monthly.merge(
        items[["year", "geo", "coicop", "item_weight"]],
        on=["year", "geo", "coicop"],
        how="left",
        validate="many_to_one",
    ).merge(
        countries[["year", "geo", "country_weight"]],
        on=["year", "geo"],
        how="left",
        validate="many_to_one",
    )

    panel = panel[
        [
            "time_period",
            "geo",
            "coicop",
            "hicp_index",
            "item_weight",
            "country_weight",
        ]
    ]
    panel.to_sql(
        "hicp_panel_data",
        conn,
        if_exists="append",
        index=False,
        chunksize=50_000,
    )
    conn.commit()
    return len(panel)


# TODO re-review this function
def core_index_series(
    frame: pd.DataFrame,
    core: str,
    parent: str,
    sub: str,
) -> pd.DataFrame:
    """Build a normalized chain-linked core index for one geography.

    The parent index is decomposed into ``sub`` and a residual ``core``
    using annual item weights and December link levels:

        I_core = L_core * (
            w_parent / w_core * I_parent / L_parent
            - w_sub / w_core * I_sub / L_sub
        )

    The first available month is used as the initial link when the prior
    December is unavailable. The result is rescaled so its 2025 average
    is 100.

    Args:
        frame: HICP rows for one geography containing ``parent`` and ``sub``.
        core: Derived COICOP code to assign to the result.
        parent: Parent COICOP code, for example ``CP04``.
        sub: Subcomponent COICOP code to remove, for example ``CP045``.

    Returns:
        Rows for the derived component with the target table columns.

    Raises:
        RuntimeError: If the result cannot be normalized to its 2025 base.
    """
    parent_rows = frame[frame["coicop"] == parent].set_index("time_period")
    sub_rows = frame[frame["coicop"] == sub].set_index("time_period")
    parent_rows = parent_rows[
        ["geo", "hicp_index", "item_weight", "country_weight"]
    ]
    sub_rows = sub_rows[["hicp_index", "item_weight"]]
    data = parent_rows.join(
        sub_rows,
        how="inner",
        lsuffix="_parent",
        rsuffix="_sub",
    )
    data = data.dropna(
        subset=[
            "hicp_index_parent",
            "hicp_index_sub",
            "item_weight_parent",
            "item_weight_sub",
        ]
    )
    data = data[data["item_weight_parent"] > data["item_weight_sub"]]
    data.index = data.index.astype(str)
    data = data.sort_index()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "time_period",
                "geo",
                "coicop",
                "hicp_index",
                "item_weight",
                "country_weight",
            ]
        )

    values = {}
    for year in sorted(data.index.str[:4].unique()):
        year_rows = data[data.index.str[:4] == year]
        prior_december = f"{int(year) - 1}-12-01"
        if prior_december in data.index and prior_december in values:
            link_period = prior_december
            core_link = values[link_period]
        else:
            link_period = year_rows.index[0]
            core_link = year_rows.loc[link_period, "hicp_index_parent"]

        parent_link = data.loc[link_period, "hicp_index_parent"]
        sub_link = data.loc[link_period, "hicp_index_sub"]
        parent_weight = year_rows.iloc[0]["item_weight_parent"]
        sub_weight = year_rows.iloc[0]["item_weight_sub"]
        core_weight = parent_weight - sub_weight

        for period, row in year_rows.iterrows():
            values[period] = core_link * (
                parent_weight * (row["hicp_index_parent"] / parent_link)
                - sub_weight * (row["hicp_index_sub"] / sub_link)
            ) / core_weight

    core_series = pd.Series(values).sort_index()
    base = core_series[core_series.index.str.startswith("2025")].mean()
    if pd.isna(base) or base == 0:
        raise RuntimeError(
            f"Cannot normalize {core} for {frame['geo'].iloc[0]}"
        )
    core_series *= 100 / base

    result = data.loc[core_series.index].copy()
    result["time_period"] = result.index
    result["coicop"] = core
    result["hicp_index"] = core_series
    result["item_weight"] = (
        result["item_weight_parent"] - result["item_weight_sub"]
    )
    return result[
        [
            "time_period",
            "geo",
            "coicop",
            "hicp_index",
            "item_weight",
            "country_weight",
        ]
    ].reset_index(drop=True)


def add_core_components(conn: sqlite3.Connection) -> int:
    """Derive and insert the chain-linked core components.

    Args:
        conn: SQLite connection containing the raw loaded panel.

    Returns:
        Number of derived rows inserted.
    """
    codes = ["CP04", "CP045", "CP07", "CP0722"]
    placeholders = ",".join("?" for _ in codes)
    frame = pd.read_sql_query(
        f"""
        SELECT time_period, geo, coicop, hicp_index, item_weight, country_weight
        FROM hicp_panel_data
        WHERE coicop IN ({placeholders})
        """,
        conn,
        params=codes,
    )

    derived = []
    for _, geo_frame in frame.groupby("geo"):
        for core, (parent, sub) in CORE_COMPONENTS.items():
            rows = core_index_series(geo_frame, core, parent, sub)
            if not rows.empty:
                derived.append(rows)

    if not derived:
        return 0

    result = pd.concat(derived, ignore_index=True)
    core_placeholders = ",".join("?" for _ in CORE_COMPONENTS)
    conn.execute(
        f"DELETE FROM hicp_panel_data WHERE coicop IN ({core_placeholders})",
        tuple(CORE_COMPONENTS),
    )
    result.to_sql(
        "hicp_panel_data",
        conn,
        if_exists="append",
        index=False,
        chunksize=50_000,
    )
    conn.commit()
    return len(result)


def main(
    db_path: str = DB_PATH,
    start_year: int = START_YEAR,
    end_year: int | None = None,
) -> None:
    """Run the Eurostat HICP ETL pipeline.

    Args:
        db_path: Destination SQLite database file.
        start_year: First calendar year to load.
        end_year: Last calendar year to load; defaults to the current year.
    """
    end_year = end_year or date.today().year
    total = 0
    with requests.Session() as session:
        session.headers["Accept"] = ACCEPT
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS hicp_panel_data")
            conn.execute(SCHEMA)
            conn.commit()
            for year in range(start_year, end_year + 1):
                count = load_year(conn, session, year)
                total += count
                logging.info("%d: %s rows", year, f"{count:,}")
            core_count = add_core_components(conn)
            total += core_count
            logging.info("Derived core components: %s rows", f"{core_count:,}")
    logging.info("Done: %s rows in %s", f"{total:,}", db_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()

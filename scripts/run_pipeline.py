"""Run the full HICP data pipeline used by the dashboard.

The pipeline runs the Eurostat ETL into ``data/hicp.db``, executes the
forecasting and pass-through models to refresh the CSV artifacts in ``data``,
and finally rebuilds the dashboard bundle in ``app/data``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hicp_etl
import hicp_forecast
from scripts.build_app_data import build_app_data

APP_DATA_DIR = ROOT / "app" / "data"


def run_pipeline(
    *,
    start_year: int = hicp_etl.START_YEAR,
    end_year: int | None = None,
) -> None:
    """Run the ETL, forecast, and dashboard data build in order."""
    hicp_etl.main(start_year=start_year, end_year=end_year)
    hicp_forecast.main()
    build_app_data(APP_DATA_DIR)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-year",
        type=int,
        default=hicp_etl.START_YEAR,
        help="First calendar year to load from Eurostat.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last calendar year to load; defaults to the current year.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline from the command line."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()

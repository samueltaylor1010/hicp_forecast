"""Human-readable labels and stable colours for the dashboard."""

from __future__ import annotations

import matplotlib

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "DE": "Germany",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PT": "Portugal",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "EA": "Euro area",
}

COMPONENT_NAMES = {
    "CP01": "Food & non-alcoholic beverages",
    "CP02": "Alcoholic beverages & tobacco",
    "CP03": "Clothing & footwear",
    "CP045": "Electricity, gas & other fuels",
    "CP04_CORE": "Housing & utilities excl. energy",
    "CP05": "Furnishings & household equipment",
    "CP06": "Health",
    "CP0722": "Fuels & lubricants for transport",
    "CP07_CORE": "Transport excl. fuels",
    "CP08": "Communications",
    "CP09": "Recreation & culture",
    "CP10": "Education",
    "CP11": "Restaurants & hotels",
    "CP12": "Miscellaneous goods & services",
    "CP13": "Personal care & social protection",
}

_QUALITATIVE = [
    matplotlib.colors.to_hex(color)
    for color in matplotlib.colormaps["tab20"].colors
]
_QUALITATIVE_EXTRA = [
    matplotlib.colors.to_hex(color)
    for color in matplotlib.colormaps["tab20b"].colors
]
_ALL_COLORS = [*_QUALITATIVE, *_QUALITATIVE_EXTRA, "#34495e", "#7f8c8d"]


def country_label(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


def component_label(code: str) -> str:
    return COMPONENT_NAMES.get(code, code)


def country_color_map(codes: list[str]) -> dict[str, str]:
    return {
        code: _ALL_COLORS[index % len(_ALL_COLORS)]
        for index, code in enumerate(codes)
    }


def component_color_map(codes: list[str]) -> dict[str, str]:
    return {
        code: _ALL_COLORS[index % len(_ALL_COLORS)]
        for index, code in enumerate(codes)
    }

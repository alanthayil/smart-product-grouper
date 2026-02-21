"""Normalize raw product records (clean text, standardize fields)."""

from __future__ import annotations

import re
from typing import TypedDict

_NON_ALNUM_RUNS = re.compile(r"[^0-9a-zA-Z.]+")
_WHITESPACE_RUNS = re.compile(r"\s+")
_NUMBER_UNIT_RUNS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|l|ml)\b",
    flags=re.IGNORECASE,
)

_UNIT_CONVERSION = {
    "g": (1.0, "g"),
    "kg": (1000.0, "g"),
    "oz": (28.349523125, "g"),
    "lb": (453.59237, "g"),
    "lbs": (453.59237, "g"),
    "ml": (1.0, "ml"),
    "l": (1000.0, "ml"),
}


class UnitInfo(TypedDict):
    unit_value: float
    unit_name: str
    unit_system: str


def _normalize_unit_tokens(text: str) -> str:
    """Ensure unit expressions have explicit spacing before text cleanup."""

    def _add_space(match: re.Match[str]) -> str:
        value, unit = match.group(1), match.group(2).lower()
        return f"{value} {unit}"

    return _NUMBER_UNIT_RUNS.sub(_add_space, text)


def _clean_text(value: object) -> str:
    """Apply baseline free-text cleaning for product descriptions."""
    text = _normalize_unit_tokens(str(value).lower().strip())
    text = _NON_ALNUM_RUNS.sub(" ", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    return text.strip()


def _extract_unit_info(description: str) -> UnitInfo | None:
    """Extract first unit mention and convert to canonical metric form."""
    match = _NUMBER_UNIT_RUNS.search(description)
    if not match:
        return None
    raw_value, raw_unit = match.group(1), match.group(2).lower()
    factor, canonical_unit = _UNIT_CONVERSION[raw_unit]
    canonical_value = round(float(raw_value) * factor, 3)
    return {
        "unit_value": canonical_value,
        "unit_name": canonical_unit,
        "unit_system": "metric",
    }


def normalize(records: list[dict]) -> list[dict]:
    """Normalize a list of raw records."""
    normalized: list[dict] = []
    for record in records:
        description = _clean_text(record.get("Description", ""))
        unit_info = _extract_unit_info(description)
        normalized_record = {
            "description": description,
            "unit_value": None,
            "unit_name": None,
            "unit_system": None,
        }
        if unit_info:
            normalized_record.update(unit_info)
        normalized.append(normalized_record)
    return normalized

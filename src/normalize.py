"""Normalize raw product records (clean text, standardize fields)."""

from __future__ import annotations

import re

_NON_ALNUM_RUNS = re.compile(r"[^0-9a-zA-Z]+")
_WHITESPACE_RUNS = re.compile(r"\s+")


def _clean_text(value: object) -> str:
    """Apply baseline free-text cleaning for product descriptions."""
    text = str(value).lower().strip()
    text = _NON_ALNUM_RUNS.sub(" ", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    return text.strip()


def normalize(records: list[dict]) -> list[dict]:
    """Normalize a list of raw records."""
    normalized: list[dict] = []
    for record in records:
        normalized.append({"description": _clean_text(record.get("Description", ""))})
    return normalized

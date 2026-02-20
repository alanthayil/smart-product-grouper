"""Tests for text normalization behavior."""

from __future__ import annotations

import pytest

from src.normalize import _clean_text, normalize


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("  WHITE   HANGING HEART  ", "white hanging heart"),
        ("Tea-Light_holder/Set,Large", "tea light holder set large"),
        ("already clean text", "already clean text"),
        ("   \t  ", ""),
    ],
)
def test_clean_text_basic_behavior(raw_text: str, expected: str) -> None:
    assert _clean_text(raw_text) == expected


def test_normalize_cleans_description_only() -> None:
    records = [
        {
            "Invoice": "ABC-001",
            "StockCode": "SKU_123",
            "Description": "  WHITE-HANGING___HEART  ",
            "Country": "United Kingdom",
        }
    ]

    normalized = normalize(records)

    assert normalized == [{"description": "white hanging heart"}]

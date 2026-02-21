"""Tests for text normalization behavior."""

from __future__ import annotations

import pytest

from src.normalize import _apply_synonyms, _clean_text, _extract_unit_info, normalize


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

    assert normalized == [
        {
            "description": "white hanging heart",
            "unit_value": None,
            "unit_name": None,
            "unit_system": None,
        }
    ]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (
            "bottle 250 ml",
            {"unit_value": 250.0, "unit_name": "ml", "unit_system": "metric"},
        ),
        (
            "powder 500 g",
            {"unit_value": 500.0, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "jar 2 oz",
            {"unit_value": 56.699, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "bag 1 lb",
            {"unit_value": 453.592, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "flour 0.5 kg",
            {"unit_value": 500.0, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "water 1 l",
            {"unit_value": 1000.0, "unit_name": "ml", "unit_system": "metric"},
        ),
    ],
)
def test_extract_unit_info_metric_and_imperial(description: str, expected: dict) -> None:
    assert _extract_unit_info(description) == expected


def test_extract_unit_info_returns_none_without_unit() -> None:
    assert _extract_unit_info("white hanging heart holder") is None


def test_normalize_formats_units_and_extracts_structured_fields() -> None:
    records = [{"Description": "Premium olive oil 1L bottle"}]

    normalized = normalize(records)

    assert normalized == [
        {
            "description": "premium olive oil 1 l bottle",
            "unit_value": 1000.0,
            "unit_name": "ml",
            "unit_system": "metric",
        }
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("steel screw set", "steel bolt set"),
        ("hex head screw", "hexagon bolt"),
        ("hex screw 2oz", "hexagon bolt 2oz"),
    ],
)
def test_apply_synonyms_for_words_and_phrases(text: str, expected: str) -> None:
    assert _apply_synonyms(text) == expected


def test_apply_synonyms_respects_word_boundaries() -> None:
    assert _apply_synonyms("hexagonalizer tool") == "hexagonalizer tool"


def test_normalize_applies_synonyms_and_preserves_unit_extraction() -> None:
    records = [{"Description": "HEX HEAD SCREW 2oz pack"}]

    normalized = normalize(records)

    assert normalized == [
        {
            "description": "hexagon bolt 2 oz pack",
            "unit_value": 56.699,
            "unit_name": "g",
            "unit_system": "metric",
        }
    ]


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("HeX-HeAd ScReW", "hex head screw"),
        ("MIXED\tCase\nITEM", "mixed case item"),
    ],
)
def test_clean_text_handles_mixed_casing_edge_cases(raw_text: str, expected: str) -> None:
    assert _clean_text(raw_text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HEX HEAD SCREW", "hexagon bolt"),
        ("hEx ScReW", "hexagon bolt"),
    ],
)
def test_apply_synonyms_handles_mixed_casing(text: str, expected: str) -> None:
    assert _apply_synonyms(text) == expected


def test_normalize_handles_mixed_casing_end_to_end() -> None:
    records = [{"Description": "HeX HeAd ScReW 2Oz"}]

    normalized = normalize(records)

    assert normalized == [
        {
            "description": "hexagon bolt 2 oz",
            "unit_value": 56.699,
            "unit_name": "g",
            "unit_system": "metric",
        }
    ]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (
            "HEX BOLT 2OZ",
            {"unit_value": 56.699, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "hex bolt 2 oz",
            {"unit_value": 56.699, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "hex bolt 2oz",
            {"unit_value": 56.699, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "hex bolt 0.50KG",
            {"unit_value": 500.0, "unit_name": "g", "unit_system": "metric"},
        ),
        (
            "hex bolt 1L",
            {"unit_value": 1000.0, "unit_name": "ml", "unit_system": "metric"},
        ),
        (
            "hex bolt 250ML",
            {"unit_value": 250.0, "unit_name": "ml", "unit_system": "metric"},
        ),
    ],
)
def test_extract_unit_info_handles_unit_variations(description: str, expected: dict) -> None:
    cleaned = _clean_text(description)
    assert _extract_unit_info(cleaned) == expected


def test_normalize_handles_weird_formatting_and_stays_deterministic() -> None:
    records = [{"Description": "  HEX---HEAD___SCREW,\n\n2OZ\t\tPACK!!!  "}]

    normalized = normalize(records)

    assert normalized == [
        {
            "description": "hexagon bolt 2 oz pack",
            "unit_value": 56.699,
            "unit_name": "g",
            "unit_system": "metric",
        }
    ]

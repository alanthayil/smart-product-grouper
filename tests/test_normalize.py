"""Tests for text normalization behavior."""

from __future__ import annotations

import pytest

from src.normalize import (
    _apply_synonyms,
    _clean_text,
    _extract_color,
    _extract_quantity_total,
    _extract_unit_info,
    _normalize_for_embedding,
    normalize,
)


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
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "SKU_123",
            "description": "white hanging heart",
            "color": "white",
            "quantity_total": None,
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
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "",
            "description": "premium olive oil 1 l bottle",
            "color": None,
            "quantity_total": None,
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
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "",
            "description": "hexagon bolt 2 oz pack",
            "color": None,
            "quantity_total": None,
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
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "",
            "description": "hexagon bolt 2 oz",
            "color": None,
            "quantity_total": None,
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
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "",
            "description": "hexagon bolt 2 oz pack",
            "color": None,
            "quantity_total": None,
            "unit_value": 56.699,
            "unit_name": "g",
            "unit_system": "metric",
        }
    ]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("blue ceramic mug", "blue"),
        ("GREY tea cup", "gray"),
        ("no explicit tone here", None),
    ],
)
def test_extract_color_returns_canonical_color_or_none(
    description: str, expected: str | None
) -> None:
    assert _extract_color(description) == expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("60", 60),
        ("6 x 10", 60),
        ("6x10", 60),
        ("pack of 72", 72),
        ("olive oil 1 l bottle", None),
        ("no quantity signal", None),
    ],
)
def test_extract_quantity_total_handles_minimal_pack_patterns(
    description: str, expected: int | None
) -> None:
    cleaned = _clean_text(description)
    assert _extract_quantity_total(cleaned) == expected


def test_normalize_preserves_stock_code_and_emits_deterministic_ids() -> None:
    records = [
        {"StockCode": "ABC123", "Description": "Sparkling water 500ml"},
        {"stock_code": "xyz-9", "Description": "Sparkling water 500 ml"},
    ]

    normalized = normalize(records)

    assert normalized[0]["stock_code"] == "ABC123"
    assert normalized[0]["row_index"] == 0
    assert normalized[0]["record_id"] == "record-0"

    assert normalized[1]["stock_code"] == "xyz-9"
    assert normalized[1]["row_index"] == 1
    assert normalized[1]["record_id"] == "record-1"


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("two seven eleven", "2 7 11"),
        ("ZERO one Twenty", "0 1 20"),
    ],
)
def test_normalize_for_embedding_converts_number_words_to_digits(
    raw_text: str, expected: str
) -> None:
    assert _normalize_for_embedding(_clean_text(raw_text)) == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("spaceboy birdcage", "space boy bird cage"),
        ("space boy bird cage", "space boy bird cage"),
    ],
)
def test_normalize_for_embedding_canonicalizes_common_token_splits(
    raw_text: str, expected: str
) -> None:
    assert _normalize_for_embedding(_clean_text(raw_text)) == expected


def test_normalize_for_embedding_drops_moderate_noise_tokens() -> None:
    cleaned = _clean_text("the mug of the set with lid and handle")
    assert _normalize_for_embedding(cleaned) == "mug set lid handle"


def test_normalize_removes_noise_tokens_but_preserves_pack_quantity_extraction() -> None:
    records = [{"Description": "Spaceboy mug pack of 72"}]

    normalized = normalize(records)

    assert normalized == [
        {
            "record_id": "record-0",
            "row_index": 0,
            "stock_code": "",
            "description": "space boy mug pack 72",
            "color": None,
            "quantity_total": 72,
            "unit_value": None,
            "unit_name": None,
            "unit_system": None,
        }
    ]


def test_normalize_toggle_disables_number_word_canonicalization() -> None:
    records = [{"Description": "Party candle two pack"}]

    normalized = normalize(records, canonicalize_number_words=False)

    assert normalized[0]["description"] == "party candle two pack"


def test_normalize_toggle_disables_token_split_canonicalization() -> None:
    records = [{"Description": "Spaceboy Birdcage"}]

    normalized = normalize(records, canonicalize_token_splits=False)

    assert normalized[0]["description"] == "spaceboy birdcage"


def test_normalize_toggle_disables_noise_token_removal() -> None:
    records = [{"Description": "the mug of the set"}]

    normalized = normalize(records, remove_noise_tokens=False)

    assert normalized[0]["description"] == "the mug of the set"


def test_normalize_toggle_disables_color_extraction() -> None:
    records = [{"Description": "Blue ceramic mug"}]

    normalized = normalize(records, extract_color=False)

    assert normalized[0]["color"] is None


def test_normalize_toggle_disables_quantity_extraction() -> None:
    records = [{"Description": "Spaceboy mug pack of 72"}]

    normalized = normalize(records, extract_quantity=False)

    assert normalized[0]["quantity_total"] is None

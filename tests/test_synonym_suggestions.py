"""Tests for unmatched-token synonym suggestion analysis."""

from __future__ import annotations

from src.synonym_suggestions import analyze_unmatched_tokens


def test_analyze_unmatched_tokens_filters_low_signal_terms() -> None:
    raw_records = [
        {"Description": "Anchor rivet 2oz pack and clamp"},
        {"Description": "The anchor with rivet 1 lb"},
        {"Description": "Anchor 250 ml pro"},
    ]

    assert analyze_unmatched_tokens(raw_records) == [
        {"token": "anchor", "count": 3},
        {"token": "rivet", "count": 2},
        {"token": "clamp", "count": 1},
        {"token": "pro", "count": 1},
    ]


def test_analyze_unmatched_tokens_excludes_known_synonym_tokens() -> None:
    raw_records = [
        {"Description": "hex head screw"},
        {"Description": "machine screw and bolt"},
        {"Description": "hexagonal fastener bolt"},
    ]

    assert analyze_unmatched_tokens(raw_records) == []


def test_analyze_unmatched_tokens_respects_limit() -> None:
    raw_records = [
        {"Description": "anchor rivet clamp pro"},
        {"Description": "anchor rivet clamp"},
        {"Description": "anchor rivet"},
    ]

    assert analyze_unmatched_tokens(raw_records, limit=2) == [
        {"token": "anchor", "count": 3},
        {"token": "rivet", "count": 3},
    ]

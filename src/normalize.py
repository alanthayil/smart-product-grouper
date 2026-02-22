"""Normalize raw product records (clean text, standardize fields)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

_NON_ALNUM_RUNS = re.compile(r"[^0-9a-zA-Z.]+")
_WHITESPACE_RUNS = re.compile(r"\s+")
_NUMBER_UNIT_RUNS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|l|ml)\b",
    flags=re.IGNORECASE,
)
_MULTIPLICATIVE_QUANTITY_RUNS = re.compile(
    r"\b(\d+)\s*x\s*(\d+)\b",
    flags=re.IGNORECASE,
)
_PACK_OF_QUANTITY_RUNS = re.compile(
    r"\bpack\s+of\s+(\d+)\b",
    flags=re.IGNORECASE,
)
_SUFFIX_QUANTITY_RUNS = re.compile(
    r"\b(\d+)\s*(pack|ct|count)\b",
    flags=re.IGNORECASE,
)
_STANDALONE_INTEGER_RUNS = re.compile(r"\b(\d+)\b")
_COLOR_WORD_RUNS = re.compile(
    r"\b(red|pink|blue|green|black|white|yellow|purple|orange|brown|gray|grey)\b",
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
_WORD_BOUNDARY = r"(?<![0-9a-zA-Z]){term}(?![0-9a-zA-Z])"
_SYNONYM_PATH = Path(__file__).resolve().parent.parent / "synonyms.yml"
_COLOR_CANONICAL_MAP = {
    "grey": "gray",
}
_NUMBER_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_NUMBER_WORD_RUNS = re.compile(
    _WORD_BOUNDARY.format(
        term="|".join(sorted((re.escape(word) for word in _NUMBER_WORD_TO_DIGIT), key=len, reverse=True))
    ),
    flags=re.IGNORECASE,
)
_EMBEDDING_TOKEN_CANONICAL_MAP = {
    "spaceboy": "space boy",
    "space boy": "space boy",
    "birdcage": "bird cage",
    "bird cage": "bird cage",
}
_EMBEDDING_TOKEN_CANONICAL_PATTERNS = [
    (
        re.compile(_WORD_BOUNDARY.format(term=re.escape(variant)), flags=re.IGNORECASE),
        canonical,
    )
    for variant, canonical in sorted(
        _EMBEDDING_TOKEN_CANONICAL_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
]
_EMBEDDING_NOISE_TOKENS = {
    "a",
    "an",
    "and",
    "for",
    "of",
    "the",
    "with",
}


class UnitInfo(TypedDict):
    unit_value: float
    unit_name: str
    unit_system: str


@lru_cache(maxsize=1)
def _load_synonym_map() -> dict[str, str]:
    """Load explicit synonym mappings from synonyms.yml."""
    if not _SYNONYM_PATH.exists():
        return {}
    loaded = json.loads(_SYNONYM_PATH.read_text(encoding="utf-8"))
    canonical_to_variants = loaded.get("canonical_to_variants", {})
    variant_to_canonical = {
        str(k).lower().strip(): str(v).lower().strip()
        for k, v in loaded.get("variant_to_canonical", {}).items()
    }
    for canonical, variants in canonical_to_variants.items():
        canonical_text = str(canonical).lower().strip()
        for variant in variants:
            variant_text = str(variant).lower().strip()
            variant_to_canonical.setdefault(variant_text, canonical_text)
    return {k: v for k, v in variant_to_canonical.items() if k and v}


def _apply_synonyms(text: str) -> str:
    """Replace phrase and word variants with canonical terms."""
    normalized_text = text
    synonym_map = _load_synonym_map()
    ordered_variants = sorted(synonym_map, key=len, reverse=True)
    for variant in ordered_variants:
        pattern = re.compile(
            _WORD_BOUNDARY.format(term=re.escape(variant)),
            flags=re.IGNORECASE,
        )
        normalized_text = pattern.sub(synonym_map[variant], normalized_text)
    normalized_text = _WHITESPACE_RUNS.sub(" ", normalized_text)
    return normalized_text.strip()


def _normalize_unit_tokens(text: str) -> str:
    """Ensure unit expressions have explicit spacing before text cleanup."""

    def _add_space(match: re.Match[str]) -> str:
        value, unit = match.group(1), match.group(2).lower()
        return f"{value} {unit}"

    return _NUMBER_UNIT_RUNS.sub(_add_space, text)


def _clean_text(value: object) -> str:
    """Apply baseline free-text cleaning for product descriptions."""
    text = str(value).lower().strip().replace("×", "x")
    text = _normalize_unit_tokens(text)
    text = _NON_ALNUM_RUNS.sub(" ", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    return text.strip()


def _canonicalize_number_words(text: str) -> str:
    """Convert basic cardinal number words to digit tokens."""
    return _NUMBER_WORD_RUNS.sub(lambda match: _NUMBER_WORD_TO_DIGIT[match.group(0).lower()], text)


def _canonicalize_common_token_splits(text: str) -> str:
    """Canonicalize known split/join variants to a stable phrase form."""
    normalized_text = text
    for pattern, canonical in _EMBEDDING_TOKEN_CANONICAL_PATTERNS:
        normalized_text = pattern.sub(canonical, normalized_text)
    normalized_text = _WHITESPACE_RUNS.sub(" ", normalized_text)
    return normalized_text.strip()


def _remove_embedding_noise_tokens(text: str) -> str:
    """Drop lightweight stopwords that do not add embedding signal."""
    filtered_tokens = [token for token in text.split() if token not in _EMBEDDING_NOISE_TOKENS]
    return " ".join(filtered_tokens)


def _normalize_for_embedding(
    text: str,
    *,
    canonicalize_number_words: bool = True,
    canonicalize_token_splits: bool = True,
    remove_noise_tokens: bool = True,
) -> str:
    """Apply deterministic normalization to maximize embedding recall."""
    normalized_text = text
    if canonicalize_number_words:
        normalized_text = _canonicalize_number_words(normalized_text)
    if canonicalize_token_splits:
        normalized_text = _canonicalize_common_token_splits(normalized_text)
    if remove_noise_tokens:
        normalized_text = _remove_embedding_noise_tokens(normalized_text)
    normalized_text = _WHITESPACE_RUNS.sub(" ", normalized_text)
    return normalized_text.strip()


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


def _extract_color(description: str) -> str | None:
    """Extract the first recognized canonical color token."""
    match = _COLOR_WORD_RUNS.search(description)
    if not match:
        return None
    raw_color = match.group(1).lower()
    return _COLOR_CANONICAL_MAP.get(raw_color, raw_color)


def _extract_quantity_total(description: str) -> int | None:
    """Extract canonical quantity totals from common pack/count expressions."""
    multiplicative_match = _MULTIPLICATIVE_QUANTITY_RUNS.search(description)
    if multiplicative_match:
        left, right = multiplicative_match.group(1), multiplicative_match.group(2)
        return int(left) * int(right)

    pack_of_match = _PACK_OF_QUANTITY_RUNS.search(description)
    if pack_of_match:
        return int(pack_of_match.group(1))

    suffix_match = _SUFFIX_QUANTITY_RUNS.search(description)
    if suffix_match:
        return int(suffix_match.group(1))

    if _NUMBER_UNIT_RUNS.search(description):
        return None

    standalone_numbers = [int(match.group(1)) for match in _STANDALONE_INTEGER_RUNS.finditer(description)]
    if len(standalone_numbers) == 1:
        return standalone_numbers[0]
    return None


def normalize(
    records: list[dict],
    *,
    canonicalize_number_words: bool = True,
    canonicalize_token_splits: bool = True,
    remove_noise_tokens: bool = True,
    extract_color: bool = True,
    extract_quantity: bool = True,
) -> list[dict]:
    """Normalize a list of raw records."""
    normalized: list[dict] = []
    for row_index, record in enumerate(records):
        base_description = _apply_synonyms(_clean_text(record.get("Description", "")))
        description = _normalize_for_embedding(
            base_description,
            canonicalize_number_words=canonicalize_number_words,
            canonicalize_token_splits=canonicalize_token_splits,
            remove_noise_tokens=remove_noise_tokens,
        )
        unit_info = _extract_unit_info(base_description)
        stock_code = str(record.get("StockCode", record.get("stock_code", ""))).strip()
        record_id = str(record.get("record_id", "")).strip() or f"record-{row_index}"
        normalized_record = {
            "record_id": record_id,
            "row_index": row_index,
            "stock_code": stock_code,
            "description": description,
            "color": None,
            "quantity_total": None,
            "unit_value": None,
            "unit_name": None,
            "unit_system": None,
        }
        if extract_color:
            color = _extract_color(base_description)
            if color:
                normalized_record["color"] = color
        if extract_quantity:
            quantity_total = _extract_quantity_total(base_description)
            if quantity_total is not None:
                normalized_record["quantity_total"] = quantity_total
        if unit_info:
            normalized_record.update(unit_info)
        normalized.append(normalized_record)
    return normalized

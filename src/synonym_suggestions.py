"""Suggest missing synonym candidates from unmatched description tokens."""

from __future__ import annotations

from collections import Counter
import re

from src.normalize import _UNIT_CONVERSION, _clean_text, _load_synonym_map

_NUMERIC_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")
_MIN_TOKEN_LENGTH = 3
_DEFAULT_LIMIT = 100
_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "set",
    "pack",
    "kit",
}


def _known_synonym_tokens() -> set[str]:
    """Build a set of known synonym/canonical tokens."""
    known_tokens: set[str] = set()
    for variant, canonical in _load_synonym_map().items():
        known_tokens.update(part for part in variant.split() if part)
        known_tokens.update(part for part in canonical.split() if part)
    return known_tokens


def _is_candidate_token(token: str) -> bool:
    """Return True when token is useful for synonym suggestions."""
    if not token or len(token) < _MIN_TOKEN_LENGTH:
        return False
    if token in _STOPWORDS:
        return False
    if token in _UNIT_CONVERSION:
        return False
    if _NUMERIC_TOKEN.match(token):
        return False
    return True


def analyze_unmatched_tokens(raw_records: list[dict], *, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Aggregate unmatched token frequencies from raw descriptions."""
    known_tokens = _known_synonym_tokens()
    token_counts: Counter[str] = Counter()

    for record in raw_records:
        cleaned = _clean_text(record.get("Description", ""))
        for token in cleaned.split():
            if not _is_candidate_token(token):
                continue
            if token in known_tokens:
                continue
            token_counts[token] += 1

    ranked = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
    top_ranked = ranked[: max(limit, 0)]
    return [{"token": token, "count": count} for token, count in top_ranked]

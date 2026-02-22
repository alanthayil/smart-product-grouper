"""Runtime configuration loader for CLI and API entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict, cast


class ClusterConflictConfig(TypedDict):
    enforce_color: bool
    enforce_quantity: bool


class ClusterConfig(TypedDict):
    similarity_threshold: float
    conflicts: ClusterConflictConfig


class NormalizeConfig(TypedDict):
    number_words: bool
    token_splits: bool
    noise_tokens: bool
    extract_color: bool
    extract_quantity: bool


class RuntimeConfig(TypedDict):
    cluster: ClusterConfig
    normalize: NormalizeConfig


_DEFAULT_CONFIG: RuntimeConfig = {
    "cluster": {
        "similarity_threshold": 0.85,
        "conflicts": {
            "enforce_color": True,
            "enforce_quantity": True,
        },
    },
    "normalize": {
        "number_words": True,
        "token_splits": True,
        "noise_tokens": True,
        "extract_color": True,
        "extract_quantity": True,
    },
}


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "defaults.json"


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_runtime_config(config_path: str | None = None) -> RuntimeConfig:
    """Load runtime config from JSON file with safe defaults."""
    configured_path = config_path or os.getenv("SMART_GROUPER_CONFIG_PATH", "").strip()
    path = Path(configured_path) if configured_path else _default_config_path()
    payload: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = cast(dict[str, object], loaded)
        except (OSError, json.JSONDecodeError):
            payload = {}

    cluster_payload = payload.get("cluster")
    cluster_map = cluster_payload if isinstance(cluster_payload, dict) else {}
    conflicts_payload = cluster_map.get("conflicts")
    conflicts_map = conflicts_payload if isinstance(conflicts_payload, dict) else {}

    normalize_payload = payload.get("normalize")
    normalize_map = normalize_payload if isinstance(normalize_payload, dict) else {}

    return {
        "cluster": {
            "similarity_threshold": _safe_float(
                cluster_map.get("similarity_threshold"),
                _DEFAULT_CONFIG["cluster"]["similarity_threshold"],
            ),
            "conflicts": {
                "enforce_color": _safe_bool(
                    conflicts_map.get("enforce_color"),
                    _DEFAULT_CONFIG["cluster"]["conflicts"]["enforce_color"],
                ),
                "enforce_quantity": _safe_bool(
                    conflicts_map.get("enforce_quantity"),
                    _DEFAULT_CONFIG["cluster"]["conflicts"]["enforce_quantity"],
                ),
            },
        },
        "normalize": {
            "number_words": _safe_bool(
                normalize_map.get("number_words"),
                _DEFAULT_CONFIG["normalize"]["number_words"],
            ),
            "token_splits": _safe_bool(
                normalize_map.get("token_splits"),
                _DEFAULT_CONFIG["normalize"]["token_splits"],
            ),
            "noise_tokens": _safe_bool(
                normalize_map.get("noise_tokens"),
                _DEFAULT_CONFIG["normalize"]["noise_tokens"],
            ),
            "extract_color": _safe_bool(
                normalize_map.get("extract_color"),
                _DEFAULT_CONFIG["normalize"]["extract_color"],
            ),
            "extract_quantity": _safe_bool(
                normalize_map.get("extract_quantity"),
                _DEFAULT_CONFIG["normalize"]["extract_quantity"],
            ),
        },
    }

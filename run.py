#!/usr/bin/env python3
"""Run the pipeline: ingest → normalize → extract → cluster → canonicalize → evaluate."""

from __future__ import annotations

import argparse
import json

from src.auto_tune import tune_similarity_threshold
from src.config import load_runtime_config
from src.ingest import ingest
from src.normalize import normalize
from src.extract import extract
from src.cluster import cluster
from src.canonicalize import canonicalize
from src.evaluate import evaluate


def _parse_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated threshold list."""
    thresholds: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("Provide at least one threshold value.")
    return thresholds


def _load_labeled_assignments(path: str) -> dict[str, object]:
    """Load labeled assignments from JSON."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("labels"), dict):
        return {str(key): value for key, value in payload["labels"].items()}
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, list):
        assignments: dict[str, object] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id", "")).strip()
            if not record_id:
                continue
            label = (
                item.get("true_cluster_id")
                if "true_cluster_id" in item
                else item.get("cluster_id")
            )
            assignments[record_id] = label
        if assignments:
            return assignments

    raise ValueError(
        "Unsupported labels format. Use JSON object mapping record_id to true_cluster_id, "
        "or {'labels': {...}}, or list entries with record_id + true_cluster_id."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run clustering pipeline with optional threshold auto-tuning."
    )
    parser.add_argument("input_path", help="Path to .xlsx input data.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help=(
            "Similarity threshold used when auto-tuning is disabled "
            "(defaults to config value)."
        ),
    )
    parser.add_argument(
        "--auto-tune-thresholds",
        action="store_true",
        help="Enable threshold sweep and pick the best threshold by F1.",
    )
    parser.add_argument(
        "--tune-thresholds",
        default="0.75,0.8,0.85,0.9,0.95",
        help="Comma-separated thresholds to sweep (default: 0.75,0.8,0.85,0.9,0.95).",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Path to JSON labeled assignments used for threshold auto-tuning.",
    )
    parser.add_argument(
        "--disable-blocking",
        action="store_true",
        help="Disable blocking and force all-pairs clustering at any input size.",
    )
    parser.add_argument(
        "--blocking-small-input-cutoff",
        type=int,
        default=300,
        help="Use all-pairs when record count is at or below this cutoff (default: 300).",
    )
    parser.add_argument(
        "--blocking-rare-token-max-frequency",
        type=int,
        default=5,
        help="Token frequency threshold used for rare-token blocking keys (default: 5).",
    )
    parser.add_argument(
        "--edge-debug",
        action="store_true",
        help="Include edge-decision debug report for top candidate pairs.",
    )
    parser.add_argument(
        "--edge-debug-top-k",
        type=int,
        default=20,
        help="Maximum number of top candidate pairs to include in edge debug report.",
    )
    args = parser.parse_args(argv)
    runtime_config = load_runtime_config()
    cluster_config = runtime_config["cluster"]
    normalize_config = runtime_config["normalize"]

    input_path = args.input_path
    raw = ingest(input_path)
    normalized = normalize(
        raw,
        canonicalize_number_words=normalize_config["number_words"],
        canonicalize_token_splits=normalize_config["token_splits"],
        remove_noise_tokens=normalize_config["noise_tokens"],
        extract_color=normalize_config["extract_color"],
        extract_quantity=normalize_config["extract_quantity"],
    )
    features = extract(normalized)

    selected_threshold = (
        float(args.similarity_threshold)
        if args.similarity_threshold is not None
        else float(cluster_config["similarity_threshold"])
    )
    tuning_summary: dict[str, object] | None = None

    if args.auto_tune_thresholds:
        if not args.labels_path:
            raise ValueError("--labels-path is required when --auto-tune-thresholds is set.")
        labeled_assignments = _load_labeled_assignments(args.labels_path)
        candidate_thresholds = _parse_thresholds(args.tune_thresholds)
        tuning_summary = tune_similarity_threshold(
            features=features,
            labeled_assignments=labeled_assignments,
            candidate_thresholds=candidate_thresholds,
        )
        selected_threshold = float(tuning_summary["best_threshold"])

    edge_debug_report: dict[str, object] = {}
    try:
        clusters = cluster(
            features,
            similarity_threshold=selected_threshold,
            enforce_color_conflict=cluster_config["conflicts"]["enforce_color"],
            enforce_quantity_conflict=cluster_config["conflicts"]["enforce_quantity"],
            enable_blocking=not args.disable_blocking,
            blocking_small_input_cutoff=args.blocking_small_input_cutoff,
            rare_token_max_frequency=args.blocking_rare_token_max_frequency,
            edge_debug=args.edge_debug,
            edge_debug_top_k=args.edge_debug_top_k,
            edge_debug_collector=edge_debug_report,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        # Backward-compatible fallback for test doubles or legacy call sites.
        clusters = cluster(features, similarity_threshold=selected_threshold)
    labels = canonicalize(clusters)
    report = evaluate(clusters, labels)
    if tuning_summary is not None:
        report["tuning"] = tuning_summary
    if args.edge_debug:
        report["edge_debug"] = edge_debug_report
    report["similarity_threshold"] = selected_threshold
    print("Report:", report)


if __name__ == "__main__":
    main()

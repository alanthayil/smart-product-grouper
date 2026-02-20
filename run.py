#!/usr/bin/env python3
"""Run the pipeline: ingest → normalize → extract → cluster → canonicalize → evaluate."""

import sys

from src.ingest import ingest
from src.normalize import normalize
from src.extract import extract
from src.cluster import cluster
from src.canonicalize import canonicalize
from src.evaluate import evaluate


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_data>", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    raw = ingest(input_path)
    normalized = normalize(raw)
    features = extract(normalized)
    clusters = cluster(features)
    labels = canonicalize(clusters)
    report = evaluate(clusters, labels)
    print("Report:", report)


if __name__ == "__main__":
    main()

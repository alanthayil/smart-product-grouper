#!/usr/bin/env python3
"""Generate evaluation markdown report from evaluation JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.reporting import render_evaluation_report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1 or len(args) > 2:
        print(
            "Usage: python generate_report.py <evaluation_json_path> [output_path]",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args[0])
    output_path = Path(args[1]) if len(args) == 2 else Path("evaluation_report.md")

    try:
        raw = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading input file: {exc}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON: {exc}", file=sys.stderr)
        return 1

    markdown = render_evaluation_report(payload)
    try:
        output_path.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"Error writing report file: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote markdown report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

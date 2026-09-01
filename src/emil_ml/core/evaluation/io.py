"""Writes the machine-readable half of an evaluation report — see
core/evaluation/__init__.py. The plotted half (plots.py) is for a human
reading the report; this is for programmatic comparison (e.g. a future
before/after diff across two training runs' metrics.json files).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    """Pretty-printed, sorted — a metrics.json is meant to be read directly
    (in the file browser, in a diff tool) as much as parsed programmatically."""
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes a flat table (e.g. YOLO's per-class precision/recall/mAP) —
    one dict per row, sharing a common set of keys as columns. A no-op
    (writes nothing) if `rows` is empty, rather than an empty/header-only
    file that looks like a bug."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

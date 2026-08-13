#!/usr/bin/env python3
"""Audit simple detection/count summaries from a safe per-image CSV fixture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "image_id",
    "reference_count",
    "true_positive",
    "false_positive",
    "false_negative",
)


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _integer(row: dict[str, str], field: str, row_number: int) -> int:
    raw = row[field].strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"row {row_number}: {field} must be >= 0, got {value}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            image_id = raw["image_id"].strip()
            if not image_id:
                raise ValueError(f"row {row_number}: image_id must be non-empty")
            if image_id in seen:
                raise ValueError(f"row {row_number}: duplicate image_id {image_id!r}")
            seen.add(image_id)
            reference_count = _integer(raw, "reference_count", row_number)
            tp = _integer(raw, "true_positive", row_number)
            fp = _integer(raw, "false_positive", row_number)
            fn = _integer(raw, "false_negative", row_number)
            if tp + fn != reference_count:
                raise ValueError(
                    f"row {row_number}: true_positive + false_negative must equal "
                    f"reference_count ({tp} + {fn} != {reference_count})"
                )
            predicted_count = tp + fp
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
            rows.append(
                {
                    "image_id": image_id,
                    "reference_count": reference_count,
                    "predicted_count": predicted_count,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "count_error": predicted_count - reference_count,
                    "absolute_count_error": abs(predicted_count - reference_count),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(row["true_positive"] for row in rows)
    total_fp = sum(row["false_positive"] for row in rows)
    total_fn = sum(row["false_negative"] for row in rows)
    n = len(rows)
    summary = {
        "schema_version": "pruscope.synthetic_metric_audit.v1",
        "n_images": n,
        "totals": {
            "reference_count": sum(row["reference_count"] for row in rows),
            "predicted_count": sum(row["predicted_count"] for row in rows),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
        },
        "micro": {
            "precision": _safe_div(total_tp, total_tp + total_fp),
            "recall": _safe_div(total_tp, total_tp + total_fn),
            "f1": _safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        },
        "macro": {
            "precision": sum(row["precision"] for row in rows) / n,
            "recall": sum(row["recall"] for row in rows) / n,
            "f1": sum(row["f1"] for row in rows) / n,
        },
        "count": {
            "mae": sum(row["absolute_count_error"] for row in rows) / n,
            "bias": sum(row["count_error"] for row in rows) / n,
        },
        "per_image": rows,
    }
    return summary


def _compare(expected: Any, actual: Any, path: str = "root") -> list[str]:
    differences: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            differences.append(
                f"{path}: keys differ; expected {sorted(expected)}, actual {sorted(actual)}"
            )
            return differences
        for key in expected:
            differences.extend(_compare(expected[key], actual[key], f"{path}.{key}"))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length differs; expected {len(expected)}, actual {len(actual)}"]
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(_compare(expected_item, actual_item, f"{path}[{index}]"))
        return differences
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            differences.append(f"{path}: expected {expected!r}, actual {actual!r}")
        return differences
    if expected != actual:
        differences.append(f"{path}: expected {expected!r}, actual {actual!r}")
    return differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and optionally verify synthetic per-image detection/count metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; existing files are not overwritten.")
    parser.add_argument("--check", type=Path, help="Expected JSON to compare against.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = load_rows(args.input)
        summary = summarize(rows)
        if args.output:
            if args.output.exists():
                raise ValueError(f"refusing to overwrite existing output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.check:
            expected = json.loads(args.check.read_text(encoding="utf-8"))
            differences = _compare(expected, summary)
            if differences:
                print("FAIL: metric audit differs from expected output", file=sys.stderr)
                for difference in differences[:20]:
                    print(f"- {difference}", file=sys.stderr)
                return 1
            print("PASS: metric audit matches expected output")
        elif not args.output:
            print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

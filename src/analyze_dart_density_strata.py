"""Density-stratified paired evaluation for direct A2 versus PruScope-DART.

Strata are defined from ground-truth fruit density only.  This keeps the
descriptive subgroup analysis independent of either detector's predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_detection_errors import greedy_matches
from src.evaluate_size_stratified import (
    box_iou,
    collect_prediction_records,
    evaluate_area_range,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--dart", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def image_metrics(record, confidence: float, iou_threshold: float) -> dict[str, float]:
    selected = record.pred_scores >= confidence
    boxes = record.pred_boxes[selected]
    matches = greedy_matches(box_iou(record.gt_boxes, boxes), iou_threshold)
    tp = len(matches)
    fp = len(boxes) - tp
    fn = len(record.gt_boxes) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    small = record.gt_areas < 32.0**2
    matched_small = sum(bool(small[gt_index]) for gt_index, _ in matches)
    return {
        "targets": float(len(record.gt_boxes)),
        "predictions": float(len(boxes)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "absolute_count_error": float(abs(len(boxes) - len(record.gt_boxes))),
        "count_bias": float(len(boxes) - len(record.gt_boxes)),
        "small_targets": float(small.sum()),
        "small_recall": matched_small / max(int(small.sum()), 1),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "images": len(rows),
        "targets": int(sum(row["targets"] for row in rows)),
        "predictions": int(sum(row["predictions"] for row in rows)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "macro_image_f1": float(np.mean([row["f1"] for row in rows])),
        "small_recall_macro": float(np.mean([row["small_recall"] for row in rows])),
        "count_mae": float(np.mean([row["absolute_count_error"] for row in rows])),
        "count_bias": float(np.mean([row["count_bias"] for row in rows])),
    }


def paired_interval(
    direct: list[dict[str, float]],
    dart: list[dict[str, float]],
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    first = np.asarray([row[metric] for row in direct], dtype=np.float64)
    second = np.asarray([row[metric] for row in dart], dtype=np.float64)
    point = float((second - first).mean())
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(first), size=len(first))
        values[index] = float((second[sample] - first[sample]).mean())
    return {
        "delta_dart_minus_direct": point,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    args = parse_args()
    image_paths, _ = load_dataset(args.data, args.split)
    direct_records = collect_prediction_records(args.direct, image_paths, args.reference_size)
    dart_records = collect_prediction_records(args.dart, image_paths, args.reference_size)
    if [row.path.resolve() for row in direct_records] != [
        row.path.resolve() for row in dart_records
    ]:
        raise RuntimeError("Direct and DART image orders do not match")

    densities = np.asarray(
        [
            len(row.gt_boxes) / max((row.width * row.height) / 1_000_000.0, 1e-9)
            for row in direct_records
        ],
        dtype=np.float64,
    )
    lower, upper = np.quantile(densities, [1.0 / 3.0, 2.0 / 3.0])
    assignments = np.where(densities <= lower, "sparse", np.where(densities <= upper, "medium", "dense"))
    direct_image = [image_metrics(row, args.confidence, args.iou) for row in direct_records]
    dart_image = [image_metrics(row, args.confidence, args.iou) for row in dart_records]

    image_rows: list[dict[str, object]] = []
    strata_payload: list[dict[str, object]] = []
    for stratum_index, stratum in enumerate(("sparse", "medium", "dense")):
        indices = np.flatnonzero(assignments == stratum)
        direct_subset_records = [direct_records[int(index)] for index in indices]
        dart_subset_records = [dart_records[int(index)] for index in indices]
        direct_subset = [direct_image[int(index)] for index in indices]
        dart_subset = [dart_image[int(index)] for index in indices]
        if not direct_subset:
            continue
        systems = {}
        for name, records, rows in (
            ("direct", direct_subset_records, direct_subset),
            ("dart", dart_subset_records, dart_subset),
        ):
            systems[name] = {
                "operating_point": aggregate(rows),
                "all": evaluate_area_range(records, 0.0, math.inf, args.max_det),
                "small": evaluate_area_range(records, 0.0, 32.0**2, args.max_det),
            }
        paired = {
            metric: paired_interval(
                direct_subset,
                dart_subset,
                metric,
                args.bootstrap,
                args.seed + stratum_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("recall", "f1", "small_recall", "absolute_count_error", "count_bias")
            )
        }
        strata_payload.append(
            {
                "stratum": stratum,
                "images": len(indices),
                "density_fruit_per_megapixel": {
                    "minimum": float(densities[indices].min()),
                    "median": float(np.median(densities[indices])),
                    "maximum": float(densities[indices].max()),
                },
                "systems": systems,
                "paired_image_metric_intervals": paired,
            }
        )
        for index in indices:
            i = int(index)
            base = {
                "image_path": str(direct_records[i].path.resolve()),
                "stratum": stratum,
                "density_fruit_per_megapixel": float(densities[i]),
            }
            for name, values in (("direct", direct_image[i]), ("dart", dart_image[i])):
                image_rows.append({**base, "system": name, **values})

    payload = {
        "data": str(args.data.resolve()),
        "split": args.split,
        "stratification": {
            "basis": "ground_truth_fruit_per_megapixel_only",
            "lower_tertile_boundary": float(lower),
            "upper_tertile_boundary": float(upper),
        },
        "operating_point": {"confidence": args.confidence, "iou": args.iou},
        "bootstrap_iterations": args.bootstrap,
        "strata": strata_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "density_stratified_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "density_image_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

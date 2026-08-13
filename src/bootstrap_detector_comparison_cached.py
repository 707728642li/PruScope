"""Fast exact image-cluster bootstrap for paired detector comparisons.

The original implementation recomputes box IoU matching inside every bootstrap
replicate. Matching is deterministic within an image, so this version caches the
per-image score/TP/FP arrays once and only repeats the global ranking step. It
retains the same image-with-replacement estimand and COCO-style interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_size_stratified import (  # noqa: E402
    IOU_THRESHOLDS,
    RECALL_THRESHOLDS,
    ImageRecord,
    box_iou,
    collect_prediction_records,
    evaluate_area_range,
    load_dataset,
    parse_named_path,
)


SIZE_RANGES = {"all": (0.0, float("inf")), "small": (0.0, 32.0**2)}
METRICS = ("AP50", "AP50_95", "AR50", "AR50_95")


@dataclass(frozen=True)
class CachedImage:
    positives: int
    scores: tuple[np.ndarray, ...]
    true_positives: tuple[np.ndarray, ...]
    false_positives: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def cache_image(
    record: ImageRecord,
    minimum_area: float,
    maximum_area: float,
    max_det: int,
) -> CachedImage:
    positives = int(
        np.count_nonzero(
            (record.gt_areas >= minimum_area) & (record.gt_areas < maximum_area)
        )
    )
    pred_order = np.argsort(-record.pred_scores, kind="stable")[:max_det]
    pred_boxes = record.pred_boxes[pred_order]
    pred_scores = record.pred_scores[pred_order]
    pred_areas = record.pred_areas[pred_order]
    gt_ignore = ~(
        (record.gt_areas >= minimum_area) & (record.gt_areas < maximum_area)
    )
    gt_order = np.argsort(gt_ignore, kind="stable")
    gt_boxes = record.gt_boxes[gt_order]
    ordered_gt_ignore = gt_ignore[gt_order]
    ious = box_iou(pred_boxes, gt_boxes)

    scores_by_iou: list[np.ndarray] = []
    tp_by_iou: list[np.ndarray] = []
    fp_by_iou: list[np.ndarray] = []
    for iou_threshold in IOU_THRESHOLDS:
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.full(len(pred_boxes), -1, dtype=np.int32)
        for pred_index in range(len(pred_boxes)):
            best_gt = -1
            best_iou = float(iou_threshold)
            for gt_index in range(len(gt_boxes)):
                if gt_matched[gt_index]:
                    continue
                if (
                    best_gt >= 0
                    and not ordered_gt_ignore[best_gt]
                    and ordered_gt_ignore[gt_index]
                ):
                    break
                overlap = float(ious[pred_index, gt_index])
                if overlap < best_iou:
                    continue
                best_iou = overlap
                best_gt = gt_index
            if best_gt >= 0:
                pred_matched[pred_index] = best_gt
                gt_matched[best_gt] = True

        matched = pred_matched >= 0
        pred_ignore = np.zeros(len(pred_boxes), dtype=bool)
        pred_ignore[matched] = ordered_gt_ignore[pred_matched[matched]]
        unmatched = ~matched
        pred_ignore[unmatched] = ~(
            (pred_areas[unmatched] >= minimum_area)
            & (pred_areas[unmatched] < maximum_area)
        )
        keep = ~pred_ignore
        scores_by_iou.append(pred_scores[keep])
        tp_by_iou.append((matched & ~pred_ignore)[keep])
        fp_by_iou.append((unmatched & ~pred_ignore)[keep])

    return CachedImage(
        positives=positives,
        scores=tuple(scores_by_iou),
        true_positives=tuple(tp_by_iou),
        false_positives=tuple(fp_by_iou),
    )


def evaluate_cached(images: list[CachedImage], indices: np.ndarray) -> dict[str, float]:
    positives = sum(images[int(index)].positives for index in indices)
    if positives == 0:
        return {metric: float("nan") for metric in METRICS}
    aps: list[float] = []
    recalls: list[float] = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        score_parts = [images[int(index)].scores[threshold_index] for index in indices]
        tp_parts = [images[int(index)].true_positives[threshold_index] for index in indices]
        fp_parts = [images[int(index)].false_positives[threshold_index] for index in indices]
        scores = np.concatenate(score_parts) if score_parts else np.empty(0)
        true_positives = np.concatenate(tp_parts).astype(np.float64)
        false_positives = np.concatenate(fp_parts).astype(np.float64)
        order = np.argsort(-scores, kind="stable")
        tp_cumulative = np.cumsum(true_positives[order])
        fp_cumulative = np.cumsum(false_positives[order])
        recall_curve = tp_cumulative / positives
        precision_curve = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1.0)
        precision_envelope = np.maximum.accumulate(precision_curve[::-1])[::-1]
        sampled_precision = np.zeros_like(RECALL_THRESHOLDS)
        for recall_index, recall_threshold in enumerate(RECALL_THRESHOLDS):
            candidates = np.flatnonzero(recall_curve >= recall_threshold)
            if len(candidates):
                sampled_precision[recall_index] = precision_envelope[candidates[0]]
        aps.append(float(sampled_precision.mean()))
        recalls.append(float(recall_curve[-1]) if len(recall_curve) else 0.0)
    return {
        "AP50": aps[0],
        "AP50_95": float(np.mean(aps)),
        "AR50": recalls[0],
        "AR50_95": float(np.mean(recalls)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    image_paths, _ = load_dataset(args.data, args.split)
    systems = dict(map(parse_named_path, args.prediction))
    if args.reference not in systems:
        raise KeyError(f"Unknown reference system: {args.reference}")
    records = {
        name: collect_prediction_records(path, image_paths, args.reference_size)
        for name, path in systems.items()
    }
    cached = {
        (name, size): [
            cache_image(record, minimum, maximum, args.max_det)
            for record in system_records
        ]
        for name, system_records in records.items()
        for size, (minimum, maximum) in SIZE_RANGES.items()
    }
    identity = np.arange(len(image_paths), dtype=np.int64)
    point_estimates: list[dict] = []
    for name, system_records in records.items():
        for size, (minimum, maximum) in SIZE_RANGES.items():
            exact = evaluate_area_range(system_records, minimum, maximum, args.max_det)
            fast = evaluate_cached(cached[(name, size)], identity)
            for metric in METRICS:
                if not np.isclose(exact[metric], fast[metric], rtol=0.0, atol=1e-12):
                    raise AssertionError(
                        f"Cache mismatch {name}/{size}/{metric}: {exact[metric]} vs {fast[metric]}"
                    )
            point_estimates.append(
                {"system": name, "size": size, **{metric: exact[metric] for metric in METRICS}}
            )
    point_lookup = {(row["system"], row["size"]): row for row in point_estimates}
    candidates = [name for name in systems if name != args.reference]
    samples = {
        (candidate, size, metric): []
        for candidate in candidates
        for size in SIZE_RANGES
        for metric in METRICS
    }
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(image_paths), size=(args.bootstrap, len(image_paths))
    )

    def evaluate_replicate(indices: np.ndarray) -> dict[tuple[str, str], dict[str, float]]:
        """Evaluate one deterministic resample; NumPy ranking releases the GIL."""
        replicate = {
            (name, size): evaluate_cached(cached[(name, size)], indices)
            for name in systems
            for size in SIZE_RANGES
        }
        return replicate

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        replicates = executor.map(evaluate_replicate, bootstrap_indices)
        for iteration, replicate in enumerate(replicates, start=1):
            for candidate in candidates:
                for size in SIZE_RANGES:
                    for metric in METRICS:
                        samples[(candidate, size, metric)].append(
                            float(
                                replicate[(candidate, size)][metric]
                                - replicate[(args.reference, size)][metric]
                            )
                        )
            if iteration % 100 == 0:
                print(f"bootstrap={iteration}/{args.bootstrap}", flush=True)

    difference_rows: list[dict] = []
    for candidate in candidates:
        for size in SIZE_RANGES:
            for metric in METRICS:
                values = np.asarray(samples[(candidate, size, metric)])
                difference_rows.append(
                    {
                        "reference": args.reference,
                        "candidate": candidate,
                        "size": size,
                        "metric": metric,
                        "difference": float(
                            point_lookup[(candidate, size)][metric]
                            - point_lookup[(args.reference, size)][metric]
                        ),
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "probability_difference_gt_zero": float(np.mean(values > 0)),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "point_estimates.csv", point_estimates)
    write_csv(args.output_dir / "paired_bootstrap_differences.csv", difference_rows)
    report = {
        "implementation": "cached_exact_image_cluster_bootstrap",
        "data": str(args.data.resolve()),
        "split": args.split,
        "images": len(image_paths),
        "reference": args.reference,
        "bootstrap_iterations": args.bootstrap,
        "workers": max(1, args.workers),
        "point_estimates": point_estimates,
        "paired_differences": difference_rows,
    }
    (args.output_dir / "bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(difference_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

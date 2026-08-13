"""Image-cluster bootstrap for detector-to-DCOH end-to-end stage metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STAGES = ("small_green", "medium_green", "mature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--exclude-zero-reference",
        action="store_true",
        help="Restrict image-level stage metrics to images containing at least one reference fruit.",
    )
    return parser.parse_args()


def macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for stage in STAGES:
        tp = np.count_nonzero((targets == stage) & (predictions == stage))
        fp = np.count_nonzero((targets != stage) & (predictions == stage))
        fn = np.count_nonzero((targets == stage) & (predictions != stage))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def metrics(rows: np.ndarray) -> dict[str, float]:
    targets = rows[:, 0]
    predictions = rows[:, 1]
    ground_truth = rows[:, 2].astype(float)
    correct_fruit = rows[:, 3].astype(float)
    detected = predictions != "no_detection"
    return {
        "image_coverage": float(np.mean(detected)),
        "image_accuracy_with_no_detection_as_error": float(np.mean(targets == predictions)),
        "image_macro_f1_with_no_detection_as_error": macro_f1(targets, predictions),
        "joint_correct_stage_recall_over_all_gt": float(correct_fruit.sum() / ground_truth.sum()),
    }


def main() -> None:
    args = parse_args()
    with args.images.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = np.asarray(
        [
            [
                row["target_stage"],
                row["predicted_stage"],
                row["ground_truth_fruit"],
                row["correctly_staged_matched_fruit"],
            ]
            for row in source
        ],
        dtype=object,
    )
    source_images = len(rows)
    if args.exclude_zero_reference:
        rows = rows[rows[:, 2].astype(float) > 0]
    if not len(rows):
        raise RuntimeError("No evaluable images remain after applying the reference filter")
    point = metrics(rows)
    rng = np.random.default_rng(args.seed)
    samples = {name: [] for name in point}
    for _ in range(args.iterations):
        sampled = rows[rng.integers(0, len(rows), size=len(rows))]
        for name, value in metrics(sampled).items():
            samples[name].append(value)
    report = {
        "images": len(rows),
        "source_images": source_images,
        "image_eligibility": (
            "at least one human-reference fruit"
            if args.exclude_zero_reference
            else "all source images"
        ),
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "results": {
            name: {
                "estimate": estimate,
                "lower_95": float(np.percentile(samples[name], 2.5)),
                "upper_95": float(np.percentile(samples[name], 97.5)),
            }
            for name, estimate in point.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Summarize strict detector-to-stage outcomes on fruit-bearing images.

The original 240-image report includes eight images with no human-reference
fruit. Stage correctness is undefined for those images, so the manuscript's
primary image-level stage analysis uses the 232 images with at least one
reference fruit and retains the 240-image result as a conservative sensitivity
analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGES = ("small_green", "medium_green", "mature")
OUTCOMES = (*STAGES, "no_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.images_csv)
    eligible = frame.loc[frame["ground_truth_fruit"] > 0].copy()
    if len(frame) != 240 or len(eligible) != 232:
        raise RuntimeError(
            f"Expected 240 source images and 232 fruit-bearing images, got "
            f"{len(frame)} and {len(eligible)}"
        )

    matrix = [
        [
            int(
                (
                    (eligible["target_stage"] == target)
                    & (eligible["predicted_stage"] == predicted)
                ).sum()
            )
            for predicted in OUTCOMES
        ]
        for target in STAGES
    ]
    detected = eligible["predicted_stage"] != "no_detection"
    image_accuracy = float(
        (eligible["target_stage"] == eligible["predicted_stage"]).mean()
    )
    stage_f1 = []
    for stage in STAGES:
        true_positive = int(
            (
                (eligible["target_stage"] == stage)
                & (eligible["predicted_stage"] == stage)
            ).sum()
        )
        false_positive = int(
            (
                (eligible["target_stage"] != stage)
                & (eligible["predicted_stage"] == stage)
            ).sum()
        )
        false_negative = int(
            (
                (eligible["target_stage"] == stage)
                & (eligible["predicted_stage"] != stage)
            ).sum()
        )
        denominator = 2 * true_positive + false_positive + false_negative
        stage_f1.append(2 * true_positive / denominator if denominator else 0.0)
    image_macro_f1 = float(sum(stage_f1) / len(stage_f1))
    matched = int(eligible["matched_fruit"].sum())
    correct_matched = int(eligible["correctly_staged_matched_fruit"].sum())
    ground_truth = int(eligible["ground_truth_fruit"].sum())

    payload = {
        "source_images": int(len(frame)),
        "fruit_bearing_images": int(len(eligible)),
        "excluded_zero_reference_images": int((frame["ground_truth_fruit"] == 0).sum()),
        "image_eligibility": "at least one human-reference fruit",
        "stages": list(STAGES),
        "outcomes": list(OUTCOMES),
        "image_confusion_with_no_detection": matrix,
        "metrics": {
            "image_coverage": float(detected.mean()),
            "image_accuracy_with_no_detection_as_error": image_accuracy,
            "image_macro_f1_with_no_detection_as_error": image_macro_f1,
            "matched_fruit": matched,
            "correctly_staged_matched_fruit": correct_matched,
            "matched_fruit_stage_accuracy": float(correct_matched / matched),
            "ground_truth_fruit": ground_truth,
            "joint_correct_stage_recall_over_all_gt": float(correct_matched / ground_truth),
        },
        "zero_reference_diagnostic": {
            "images_with_no_detection": int(
                (
                    (frame["ground_truth_fruit"] == 0)
                    & (frame["predicted_stage"] == "no_detection")
                ).sum()
            ),
            "images_with_false_positive_detections": int(
                (
                    (frame["ground_truth_fruit"] == 0)
                    & (frame["predicted_fruit"] > 0)
                ).sum()
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

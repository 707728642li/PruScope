"""Match frozen DART proposals to protected-safe human annotations."""

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

from pruscope import DART_METADATA_NAMES, proposal_metadata_vector
from src.evaluate_size_stratified import box_iou, load_yolo_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--negative-iou", type=float, default=0.20)
    parser.add_argument("--hard-negative-confidence", type=float, default=0.02)
    parser.add_argument("--max-negatives-per-image", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if not 0 <= args.negative_iou < args.positive_iou <= 1:
        raise ValueError("Require 0 <= negative_iou < positive_iou <= 1")
    eligible = {row["task_id"]: row for row in read_csv(args.eligibility)}
    stages = {row["image_id"]: row for row in read_csv(args.stage_manifest)}
    if set(eligible) - set(stages):
        raise KeyError(f"Missing stage rows: {sorted(set(eligible) - set(stages))[:3]}")
    proposal_rows = [
        json.loads(line)
        for line in args.proposals.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proposals_by_id = {Path(row["image_path"]).stem: row for row in proposal_rows}
    if set(eligible) != set(proposals_by_id):
        raise ValueError(
            f"Eligibility/proposal ID mismatch: {len(eligible)} vs {len(proposals_by_id)}"
        )

    output_rows: list[dict] = []
    summary = {"positive": 0, "negative": 0, "ignored": 0}
    rng = np.random.default_rng(args.seed)
    for task_id in sorted(eligible):
        record = proposals_by_id[task_id]
        stage_row = stages[task_id]
        width, height = int(record["width"]), int(record["height"])
        gt_boxes, _ = load_yolo_boxes(Path(stage_row["label_path"]), width, height)
        candidates = record.get("dart_proposals", [])
        candidate_boxes = np.asarray(
            [item["xyxy"] for item in candidates], dtype=np.float32
        ).reshape(-1, 4)
        overlaps = box_iou(candidate_boxes, gt_boxes)
        if len(gt_boxes):
            best_gt = overlaps.argmax(axis=1) if len(candidate_boxes) else np.empty(0, int)
            best_iou = overlaps.max(axis=1) if len(candidate_boxes) else np.empty(0)
        else:
            best_gt = np.full(len(candidate_boxes), -1, dtype=int)
            best_iou = np.zeros(len(candidate_boxes), dtype=float)
        negative_indices = np.flatnonzero(
            (best_iou < args.negative_iou)
            & (
                np.asarray([float(item["confidence"]) for item in candidates])
                >= args.hard_negative_confidence
            )
        )
        if len(negative_indices) > args.max_negatives_per_image:
            scores = np.asarray([float(candidates[index]["confidence"]) for index in negative_indices])
            jitter = rng.uniform(0.0, 1e-9, len(scores))
            negative_indices = negative_indices[np.argsort(-(scores + jitter))[: args.max_negatives_per_image]]
        retained_negatives = set(map(int, negative_indices))
        image_density = len(candidates)
        for candidate_index, (candidate, overlap, gt_index) in enumerate(
            zip(candidates, best_iou, best_gt, strict=True)
        ):
            if overlap >= args.positive_iou:
                target = 1
                summary["positive"] += 1
            elif candidate_index in retained_negatives:
                target = 0
                summary["negative"] += 1
            else:
                summary["ignored"] += 1
                continue
            metadata = proposal_metadata_vector(candidate, width, height, image_density)
            gt_box = (
                [round(float(value), 4) for value in gt_boxes[int(gt_index)]]
                if target
                else [math.nan] * 4
            )
            output_rows.append(
                {
                    "task_id": task_id,
                    "stage": eligible[task_id]["stage"],
                    "image_path": str(Path(record["image_path"]).resolve()),
                    "width": width,
                    "height": height,
                    "proposal_index": candidate_index,
                    "proposal_box": [round(float(value), 4) for value in candidate["xyxy"]],
                    "gt_box": gt_box,
                    "target": target,
                    "match_iou": round(float(overlap), 8),
                    "metadata": [round(float(value), 8) for value in metadata],
                    "base_confidence": float(candidate["confidence"]),
                    "reference_area": (
                        (float(candidate["xyxy"][2]) - float(candidate["xyxy"][0]))
                        * (float(candidate["xyxy"][3]) - float(candidate["xyxy"][1]))
                        / (width * height)
                        * 1024**2
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    report = {
        "images": len(eligible),
        "rows": len(output_rows),
        **summary,
        "positive_iou": args.positive_iou,
        "negative_iou": args.negative_iou,
        "hard_negative_confidence": args.hard_negative_confidence,
        "metadata_names": DART_METADATA_NAMES,
        "source_proposals": str(args.proposals.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

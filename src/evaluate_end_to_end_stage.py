"""Evaluate detector-plus-DCOH stage predictions, counting missed fruit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_stage_ordinal import STAGE_NAMES, stage_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-splits", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Fail closed unless this many split rows are evaluated",
    )
    return parser.parse_args()


def load_yolo_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if not path.exists():
        return np.empty((0, 4), dtype=np.float32)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, xc, yc, bw, bh = map(float, fields[:5])
        boxes.append(
            [
                (xc - bw / 2.0) * width,
                (yc - bh / 2.0) * height,
                (xc + bw / 2.0) * width,
                (yc + bh / 2.0) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.clip(boxes1[:, 2:] - boxes1[:, :2], 0.0, None), axis=1)
    area2 = np.prod(np.clip(boxes2[:, 2:] - boxes2[:, :2], 0.0, None), axis=1)
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy_matches(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    overlaps = box_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    matches = []
    for pred_index in np.argsort(-pred_scores, kind="stable"):
        if not len(gt_boxes):
            break
        candidates = np.argsort(-overlaps[pred_index], kind="stable")
        for gt_index in candidates:
            gt_index = int(gt_index)
            if overlaps[pred_index, gt_index] < threshold:
                break
            if gt_index not in matched_gt:
                matched_gt.add(gt_index)
                matches.append((gt_index, int(pred_index)))
                break
    return matches


def main() -> None:
    args = parse_args()
    with args.stage_splits.open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if args.expected_images is not None and len(split_rows) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} {args.split} images, found {len(split_rows)}"
        )
    prediction_records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_path = {
        str(Path(record["image_path"]).resolve()).casefold(): record
        for record in prediction_records
    }
    matched_targets: list[int] = []
    matched_predictions: list[int] = []
    fruit_confusion_with_miss = np.zeros((3, 4), dtype=np.int64)
    image_confusion_with_miss = np.zeros((3, 4), dtype=np.int64)
    total_gt = total_predictions = total_matched = total_correct = 0
    image_absolute_count_errors = []
    image_targets: list[int] = []
    image_predictions: list[int] = []
    image_ids: list[str] = []
    image_rows: list[dict[str, object]] = []
    for row in split_rows:
        key = str(Path(row["source_image"]).resolve()).casefold()
        if key not in by_path:
            raise KeyError(f"Missing prediction record for {row['source_image']}")
        record = by_path[key]
        width, height = int(record["width"]), int(record["height"])
        gt_boxes = load_yolo_boxes(Path(row["label_path"]), width, height)
        predictions = record.get("predictions", [])
        pred_boxes = np.asarray(
            [prediction["xyxy"] for prediction in predictions], dtype=np.float32
        ).reshape(-1, 4)
        pred_scores = np.asarray(
            [prediction["confidence"] for prediction in predictions], dtype=np.float32
        )
        pred_stages = np.asarray(
            [STAGE_NAMES.index(prediction["stage"]) for prediction in predictions],
            dtype=np.int64,
        )
        target_stage = int(row["stage_index"])
        matches = greedy_matches(gt_boxes, pred_boxes, pred_scores, args.iou)
        matched_gt = {gt_index for gt_index, _ in matches}
        for gt_index, pred_index in matches:
            predicted_stage = int(pred_stages[pred_index])
            matched_targets.append(target_stage)
            matched_predictions.append(predicted_stage)
            fruit_confusion_with_miss[target_stage, predicted_stage] += 1
            total_correct += predicted_stage == target_stage
        fruit_confusion_with_miss[target_stage, 3] += len(gt_boxes) - len(matched_gt)
        image_ids.append(row.get("image_id", Path(row["source_image"]).stem))
        image_targets.append(target_stage)
        if predictions:
            mean_probability = np.mean(
                [
                    [prediction["stage_probabilities"][name] for name in STAGE_NAMES]
                    for prediction in predictions
                ],
                axis=0,
            )
            predicted_image_stage = int(mean_probability.argmax())
            image_confusion_with_miss[target_stage, predicted_image_stage] += 1
            image_predictions.append(predicted_image_stage)
        else:
            image_confusion_with_miss[target_stage, 3] += 1
            image_predictions.append(3)
        total_gt += len(gt_boxes)
        total_predictions += len(predictions)
        total_matched += len(matches)
        image_absolute_count_errors.append(abs(len(predictions) - len(gt_boxes)))
        image_rows.append(
            {
                "image_id": image_ids[-1],
                "target_stage": STAGE_NAMES[target_stage],
                "predicted_stage": (
                    STAGE_NAMES[image_predictions[-1]]
                    if image_predictions[-1] < len(STAGE_NAMES)
                    else "no_detection"
                ),
                "ground_truth_fruit": len(gt_boxes),
                "predicted_fruit": len(predictions),
                "matched_fruit": len(matches),
                "correctly_staged_matched_fruit": sum(
                    int(pred_stages[pred_index]) == target_stage
                    for _, pred_index in matches
                ),
            }
        )
    matched_targets_array = np.asarray(matched_targets, dtype=np.int64)
    matched_predictions_array = np.asarray(matched_predictions, dtype=np.int64)
    image_targets_array = np.asarray(image_targets, dtype=np.int64)
    image_predictions_array = np.asarray(image_predictions, dtype=np.int64)
    detected_mask = image_predictions_array < len(STAGE_NAMES)
    image_stage_metrics = (
        stage_metrics(image_targets_array[detected_mask], image_predictions_array[detected_mask])
        if np.any(detected_mask)
        else None
    )
    image_accuracy_with_no_detection_as_error = float(
        np.mean(image_targets_array == image_predictions_array)
    )
    report = {
        "split": args.split,
        "iou_threshold": args.iou,
        "images": len(split_rows),
        "ground_truth_fruit": total_gt,
        "predicted_fruit": total_predictions,
        "matched_fruit": total_matched,
        "detection_recall_at_iou": total_matched / max(total_gt, 1),
        "stage_accuracy_on_matched_fruit": total_correct / max(total_matched, 1),
        "joint_correct_stage_recall_over_all_gt": total_correct / max(total_gt, 1),
        "count_mae_per_image": float(np.mean(image_absolute_count_errors)),
        "images_with_at_least_one_detection": int(np.count_nonzero(detected_mask)),
        "image_coverage": float(np.mean(detected_mask)),
        "image_stage_accuracy_with_no_detection_as_error": image_accuracy_with_no_detection_as_error,
        "image_stage_metrics_conditional_on_detection": image_stage_metrics,
        "matched_fruit_stage_metrics": (
            stage_metrics(matched_targets_array, matched_predictions_array)
            if len(matched_targets_array)
            else None
        ),
        "fruit_confusion_columns": STAGE_NAMES + ["missed_detection"],
        "fruit_confusion_with_miss": fruit_confusion_with_miss.tolist(),
        "image_confusion_columns": STAGE_NAMES + ["no_detection"],
        "image_confusion_with_miss": image_confusion_with_miss.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    image_csv = args.output.with_name(args.output.stem + "_images.csv")
    with image_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

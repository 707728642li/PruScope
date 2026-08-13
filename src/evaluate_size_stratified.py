"""Evaluate YOLO detectors with COCO-style object-size strata.

The evaluator uses normalized bounding-box area projected onto a square reference
canvas (1024 by default). This makes the COCO small/medium/large thresholds
reproducible across source images with different resolutions and aspect ratios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import register_pruscope_modules

register_pruscope_modules()

from ultralytics import RTDETR, YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    gt_boxes: np.ndarray
    gt_areas: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_areas: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data YAML")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=WEIGHTS",
        help="Repeat for each model to compare",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        metavar="NAME=JSONL",
        help="Repeat for precomputed PruScope GLAF prediction files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save each weights model's detections as reusable JSONL",
    )
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name:
        raise ValueError(f"Empty model name in: {value}")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path


def resolve_source(value: str | list[str], dataset_root: Path, yaml_dir: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    image_paths: list[Path] = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            root_candidate = dataset_root / path
            yaml_candidate = yaml_dir / path
            path = root_candidate if root_candidate.exists() else yaml_candidate
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = dataset_root / image_path
                image_paths.append(image_path)
        elif path.is_dir():
            image_paths.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(path)
        else:
            raise FileNotFoundError(f"Could not resolve image source: {item}")
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return image_paths


def load_dataset(data_yaml: Path, split: str) -> tuple[list[Path], Path]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")
    yaml_dir = data_yaml.resolve().parent
    dataset_root = Path(payload.get("path", yaml_dir))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_dir / dataset_root).resolve()
    images = resolve_source(payload[split], dataset_root, yaml_dir)
    return images, dataset_root


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(f"Image path has no 'images' directory: {image_path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_yolo_boxes(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    areas: list[float] = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, xc, yc, bw, bh = map(float, fields[:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append([x1, y1, x2, y2])
        areas.append(max(0.0, bw) * max(0.0, bh))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(areas, dtype=np.float32)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.clip(boxes1[:, 2:] - boxes1[:, :2], 0.0, None), axis=1)
    area2 = np.prod(np.clip(boxes2[:, 2:] - boxes2[:, :2], 0.0, None), axis=1)
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def collect_records(
    model_name: str,
    model_path: Path,
    image_paths: list[Path],
    imgsz: int,
    reference_size: int,
    batch: int,
    device: str,
    conf: float,
    nms_iou: float,
    max_det: int,
) -> list[ImageRecord]:
    model_class = RTDETR if model_name.lower().startswith("rtdetr") else YOLO
    model = model_class(str(model_path))
    records: list[ImageRecord] = []
    reference_area = float(reference_size * reference_size)
    # Bound each predictor call to one requested batch. Ultralytics' image-list
    # loader retained decoded high-resolution sources even with stream=True in
    # this workload, causing progressive >40 GB host-memory growth. Re-entering
    # predict at a small, explicit call boundary releases the loader and Results
    # tensors before the next chunk while preserving the exact model settings.
    chunk_size = max(1, int(batch))
    for chunk_start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[chunk_start : chunk_start + chunk_size]
        chunk_results = model.predict(
            source=[str(path) for path in chunk_paths],
            imgsz=imgsz,
            batch=min(chunk_size, len(chunk_paths)),
            device=device,
            conf=conf,
            iou=nms_iou,
            max_det=max_det,
            verbose=False,
            stream=False,
        )
        if len(chunk_results) != len(chunk_paths):
            raise RuntimeError(
                f"Expected {len(chunk_paths)} chunk results, "
                f"received {len(chunk_results)}"
            )
        for image_path, result in zip(chunk_paths, chunk_results, strict=True):
            height, width = result.orig_shape
            gt_boxes, gt_area_fraction = load_yolo_boxes(
                image_to_label_path(image_path), width, height
            )
            if result.boxes is None or len(result.boxes) == 0:
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_scores = np.empty(0, dtype=np.float32)
                pred_area_fraction = np.empty(0, dtype=np.float32)
            else:
                pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                pred_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                wh = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None)
                pred_area_fraction = (wh[:, 0] * wh[:, 1]) / float(width * height)
            records.append(
                ImageRecord(
                    path=image_path,
                    width=width,
                    height=height,
                    gt_boxes=gt_boxes,
                    gt_areas=gt_area_fraction * reference_area,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_areas=pred_area_fraction * reference_area,
                )
            )
        del chunk_results
    if len(records) != len(image_paths):
        raise RuntimeError(
            f"Expected {len(image_paths)} results, received {len(records)}"
        )
    return records


def collect_prediction_records(
    prediction_path: Path,
    image_paths: list[Path],
    reference_size: int,
) -> list[ImageRecord]:
    """Load precomputed GLAF predictions into the same evaluation records."""
    payloads = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_path = {
        str(Path(record["image_path"]).resolve()).casefold(): record for record in payloads
    }
    records: list[ImageRecord] = []
    reference_area = float(reference_size * reference_size)
    for image_path in image_paths:
        key = str(image_path.resolve()).casefold()
        if key not in by_path:
            raise KeyError(f"No prediction for {image_path} in {prediction_path}")
        record = by_path[key]
        width, height = int(record["width"]), int(record["height"])
        gt_boxes, gt_area_fraction = load_yolo_boxes(
            image_to_label_path(image_path), width, height
        )
        predictions = record.get("predictions", [])
        pred_boxes = np.asarray(
            [item["xyxy"] for item in predictions], dtype=np.float32
        ).reshape(-1, 4)
        pred_scores = np.asarray(
            [item["confidence"] for item in predictions], dtype=np.float32
        )
        if len(pred_boxes):
            wh = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None)
            pred_area_fraction = (wh[:, 0] * wh[:, 1]) / float(width * height)
        else:
            pred_area_fraction = np.empty(0, dtype=np.float32)
        records.append(
            ImageRecord(
                path=image_path,
                width=width,
                height=height,
                gt_boxes=gt_boxes,
                gt_areas=gt_area_fraction * reference_area,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_areas=pred_area_fraction * reference_area,
            )
        )
    if len(by_path) != len(image_paths):
        print(
            f"WARNING {prediction_path} has {len(by_path)} records for "
            f"{len(image_paths)} requested images"
        )
    return records


def save_prediction_records(path: Path, records: list[ImageRecord]) -> None:
    """Save predictions in the JSONL schema accepted by --prediction."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "image_path": str(record.path.resolve()),
                "width": record.width,
                "height": record.height,
                "predictions": [
                    {
                        "xyxy": [round(float(value), 6) for value in box],
                        "confidence": round(float(score), 8),
                    }
                    for box, score in zip(
                        record.pred_boxes, record.pred_scores, strict=True
                    )
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def evaluate_area_range(
    records: list[ImageRecord],
    minimum_area: float,
    maximum_area: float,
    max_det: int,
) -> dict[str, float | int]:
    positives = int(
        sum(
            np.count_nonzero((record.gt_areas >= minimum_area) & (record.gt_areas < maximum_area))
            for record in records
        )
    )
    if positives == 0:
        return {
            "targets": 0,
            "AP50": math.nan,
            "AP50_95": math.nan,
            "AR50": math.nan,
            "AR50_95": math.nan,
            "best_F1": math.nan,
            "precision_at_best_F1": math.nan,
            "recall_at_best_F1": math.nan,
            "confidence_at_best_F1": math.nan,
        }

    aps: list[float] = []
    recalls: list[float] = []
    f1_summary: dict[str, float] | None = None
    for threshold_index, iou_threshold in enumerate(IOU_THRESHOLDS):
        all_scores: list[np.ndarray] = []
        all_true_positives: list[np.ndarray] = []
        all_false_positives: list[np.ndarray] = []

        for record in records:
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
            true_positive = matched & ~pred_ignore
            false_positive = unmatched & ~pred_ignore
            keep = ~pred_ignore
            all_scores.append(pred_scores[keep])
            all_true_positives.append(true_positive[keep])
            all_false_positives.append(false_positive[keep])

        scores = np.concatenate(all_scores) if all_scores else np.empty(0)
        true_positives = (
            np.concatenate(all_true_positives).astype(np.float64)
            if all_true_positives
            else np.empty(0)
        )
        false_positives = (
            np.concatenate(all_false_positives).astype(np.float64)
            if all_false_positives
            else np.empty(0)
        )
        order = np.argsort(-scores, kind="stable")
        scores = scores[order]
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

        if threshold_index == 0:
            f1 = np.divide(
                2.0 * precision_curve * recall_curve,
                precision_curve + recall_curve,
                out=np.zeros_like(precision_curve),
                where=(precision_curve + recall_curve) > 0,
            )
            if len(f1):
                best_index = int(np.argmax(f1))
                f1_summary = {
                    "best_F1": float(f1[best_index]),
                    "precision_at_best_F1": float(precision_curve[best_index]),
                    "recall_at_best_F1": float(recall_curve[best_index]),
                    "confidence_at_best_F1": float(scores[best_index]),
                }
            else:
                f1_summary = {
                    "best_F1": 0.0,
                    "precision_at_best_F1": 0.0,
                    "recall_at_best_F1": 0.0,
                    "confidence_at_best_F1": math.nan,
                }

    assert f1_summary is not None
    return {
        "targets": positives,
        "AP50": aps[0],
        "AP50_95": float(np.mean(aps)),
        "AR50": recalls[0],
        "AR50_95": float(np.mean(recalls)),
        **f1_summary,
    }


def main() -> None:
    args = parse_args()
    if not args.model and not args.prediction:
        raise ValueError("At least one --model or --prediction is required")
    visible_devices = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if str(args.device) != "cpu" and (not visible_devices or "," in visible_devices):
        raise RuntimeError(
            "Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES and use --device 0"
        )
    image_paths, dataset_root = load_dataset(args.data, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size_ranges = {
        "all": (0.0, math.inf),
        "small": (0.0, 32.0**2),
        "medium": (32.0**2, 96.0**2),
        "large": (96.0**2, math.inf),
    }
    output_rows: list[dict[str, object]] = []
    model_metadata: list[dict[str, str]] = []

    for model_spec in args.model or []:
        model_name, model_path = parse_named_path(model_spec)
        print(f"Evaluating {model_name}: {model_path.resolve()}")
        records = collect_records(
            model_name=model_name,
            model_path=model_path,
            image_paths=image_paths,
            imgsz=args.imgsz,
            reference_size=args.reference_size,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
        )
        if args.save_predictions:
            prediction_path = args.output_dir / f"{model_name}_predictions.jsonl"
            save_prediction_records(prediction_path, records)
            print(f"  predictions={prediction_path.resolve()}")
        model_metadata.append(
            {"name": model_name, "kind": "weights", "path": str(model_path.resolve())}
        )
        for size_name, (minimum_area, maximum_area) in size_ranges.items():
            metrics = evaluate_area_range(records, minimum_area, maximum_area, args.max_det)
            row: dict[str, object] = {
                "model": model_name,
                "size": size_name,
                **metrics,
            }
            output_rows.append(row)
            print(
                f"  {size_name:6s} n={metrics['targets']:4d} "
                f"AP50={metrics['AP50']:.4f} AP50-95={metrics['AP50_95']:.4f} "
                f"AR50={metrics['AR50']:.4f}"
            )

    for prediction_spec in args.prediction or []:
        model_name, prediction_path = parse_named_path(prediction_spec)
        print(f"Evaluating {model_name}: {prediction_path.resolve()}")
        records = collect_prediction_records(
            prediction_path=prediction_path,
            image_paths=image_paths,
            reference_size=args.reference_size,
        )
        model_metadata.append(
            {
                "name": model_name,
                "kind": "glaf_predictions",
                "path": str(prediction_path.resolve()),
            }
        )
        for size_name, (minimum_area, maximum_area) in size_ranges.items():
            metrics = evaluate_area_range(records, minimum_area, maximum_area, args.max_det)
            row = {"model": model_name, "size": size_name, **metrics}
            output_rows.append(row)
            print(
                f"  {size_name:6s} n={metrics['targets']:4d} "
                f"AP50={metrics['AP50']:.4f} AP50-95={metrics['AP50_95']:.4f} "
                f"AR50={metrics['AR50']:.4f}"
            )

    csv_path = args.output_dir / "size_stratified_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "data_yaml": str(args.data.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "split": args.split,
        "images": len(image_paths),
        "models": model_metadata,
        "settings": {
            "imgsz": args.imgsz,
            "reference_size": args.reference_size,
            "area_definition": "normalized_bbox_area * reference_size^2",
            "small": "area < 32^2",
            "medium": "32^2 <= area < 96^2",
            "large": "area >= 96^2",
            "iou_thresholds": [round(float(value), 2) for value in IOU_THRESHOLDS],
            "confidence_floor": args.conf,
            "nms_iou": args.nms_iou,
            "max_detections_per_image": args.max_det,
        },
        "results": output_rows,
    }
    json_path = args.output_dir / "size_stratified_metrics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CSV: {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()

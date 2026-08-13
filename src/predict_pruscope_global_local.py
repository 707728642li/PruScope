"""Run PruScope global-local adaptive fusion (GLAF) inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import (
    adaptive_stream_weight,
    cluster_multiview_proposals,
    generate_overlapping_tiles,
    register_pruscope_modules,
    size_aware_weighted_box_fusion,
    tile_edge_reliability,
)

register_pruscope_modules()

from ultralytics import YOLO  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-split",
        help="Optional split filter when --source is a CSV with a split column",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prediction-conf", type=float, default=0.001)
    parser.add_argument("--final-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument(
        "--local-final-conf",
        type=float,
        help="Optional stricter confidence floor for the local stream",
    )
    parser.add_argument(
        "--local-max-reference-area",
        type=float,
        default=math.inf,
        help="Keep local boxes below this area on a 1024x1024 reference canvas",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional deterministic limit after path sorting, for efficiency probes",
    )
    parser.add_argument(
        "--save-global-predictions",
        action="store_true",
        help=(
            "Store the full-image stream alongside GLAF predictions so a "
            "density gate can be selected on validation data without rerunning inference"
        ),
    )
    parser.add_argument(
        "--save-dart-proposals",
        action="store_true",
        help="Store multiview cluster statistics required by the DART tail",
    )
    parser.add_argument(
        "--density-threshold",
        type=int,
        default=0,
        help=(
            "Run local tiles only when the global detection count reaches this value; "
            "0 preserves always-GLAF behavior"
        ),
    )
    parser.add_argument(
        "--scene-gate-confidence",
        type=float,
        help="Confidence used to count global candidates for the scale-density gate",
    )
    parser.add_argument(
        "--scene-gate-count",
        type=int,
        help="Minimum confident global candidates required by the scale-density gate",
    )
    parser.add_argument(
        "--scene-gate-max-median-area",
        type=float,
        help=(
            "Maximum median global-box area on a 1024x1024 reference canvas; "
            "set all three --scene-gate-* arguments to enable the frozen gate"
        ),
    )
    return parser.parse_args()


def resolve_images(source: Path, split: str | None = None) -> list[Path]:
    if source.is_dir():
        images = sorted(
            path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
    elif source.is_file() and source.suffix.lower() == ".txt":
        images = [
            Path(line.strip())
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif source.is_file() and source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        path_column = (
            "source_image" if rows and "source_image" in rows[0]
            else "path" if rows and "path" in rows[0]
            else None
        )
        if path_column is None:
            raise ValueError(f"CSV must contain source_image or path: {source}")
        if split is not None:
            if "split" not in rows[0]:
                raise ValueError(f"CSV has no split column: {source}")
            rows = [row for row in rows if row["split"] == split]
        images = [Path(row[path_column]) for row in rows]
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        images = [source]
    else:
        raise FileNotFoundError(f"Could not resolve image source: {source}")
    missing = [path for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return images


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode {path}")
    return image


def result_tensors(result) -> tuple[torch.Tensor, torch.Tensor]:
    if result.boxes is None or len(result.boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,))
    return (
        result.boxes.xyxy.detach().cpu().float(),
        result.boxes.conf.detach().cpu().float(),
    )


def serialize_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    observations: torch.Tensor | None = None,
) -> list[dict]:
    """Convert prediction tensors to the JSONL schema used by the evaluator."""
    if observations is None:
        observations = torch.ones(len(boxes), dtype=torch.long)
    return [
        {
            "xyxy": [round(float(value), 3) for value in box],
            "confidence": round(float(score), 6),
            "observations": int(count),
        }
        for box, score, count in zip(boxes, scores, observations, strict=True)
    ]


def predict_one(model: YOLO, image_path: Path, args: argparse.Namespace) -> dict:
    image = read_image(image_path)
    image_height, image_width = image.shape[:2]
    if str(args.device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    full_result = model.predict(
        source=image,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.prediction_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        verbose=False,
    )[0]
    if str(args.device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    global_elapsed = time.perf_counter() - started
    global_peak_gpu_memory_mib = (
        float(torch.cuda.max_memory_allocated()) / float(1024**2)
        if str(args.device) != "cpu" and torch.cuda.is_available()
        else None
    )
    full_boxes, full_scores = result_tensors(full_result)
    scene_gate_count = None
    scene_gate_median_area = None
    if args.scene_gate_confidence is not None:
        confident = full_scores >= args.scene_gate_confidence
        scene_gate_count = int(confident.sum())
        if scene_gate_count:
            confident_boxes = full_boxes[confident]
            confident_wh = (confident_boxes[:, 2:] - confident_boxes[:, :2]).clamp_min(0.0)
            reference_areas = (
                confident_wh[:, 0]
                * confident_wh[:, 1]
                / float(image_width * image_height)
                * float(1024**2)
            )
            scene_gate_median_area = float(reference_areas.median())
        else:
            scene_gate_median_area = math.inf
        use_local_stream = (
            scene_gate_count >= args.scene_gate_count
            and scene_gate_median_area <= args.scene_gate_max_median_area
        )
    else:
        use_local_stream = len(full_boxes) >= args.density_threshold
    all_boxes = [full_boxes]
    all_scores = [full_scores]
    all_reliability = [
        adaptive_stream_weight(
            full_boxes, image_width, image_height, stream="global"
        )
    ]
    all_streams = [torch.zeros(len(full_boxes), dtype=torch.long)]

    tiles = (
        generate_overlapping_tiles(
            image_width,
            image_height,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
        )
        if use_local_stream
        else []
    )
    if len(tiles) == 1 and tiles[0].width == image_width and tiles[0].height == image_height:
        tiles = []
    tile_arrays = [image[tile.y1 : tile.y2, tile.x1 : tile.x2] for tile in tiles]
    tile_results = (
        model.predict(
            source=tile_arrays,
            imgsz=args.imgsz,
            batch=min(args.batch, len(tile_arrays)),
            device=args.device,
            conf=args.prediction_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            verbose=False,
        )
        if tile_arrays
        else []
    )
    for tile, result in zip(tiles, tile_results, strict=True):
        local_boxes, local_scores = result_tensors(result)
        if len(local_boxes):
            local_wh = (local_boxes[:, 2:] - local_boxes[:, :2]).clamp_min(0.0)
            local_reference_area = (
                local_wh[:, 0]
                * local_wh[:, 1]
                / float(image_width * image_height)
                * float(1024**2)
            )
            local_confidence_floor = (
                args.local_final_conf
                if args.local_final_conf is not None
                else args.prediction_conf
            )
            local_keep = (local_scores >= local_confidence_floor) & (
                local_reference_area <= args.local_max_reference_area
            )
            local_boxes = local_boxes[local_keep]
            local_scores = local_scores[local_keep]
        edge_weight = tile_edge_reliability(
            local_boxes, tile, image_width, image_height
        )
        if len(local_boxes):
            local_boxes[:, (0, 2)] += tile.x1
            local_boxes[:, (1, 3)] += tile.y1
        scale_weight = adaptive_stream_weight(
            local_boxes, image_width, image_height, stream="local"
        )
        all_boxes.append(local_boxes)
        all_scores.append(local_scores)
        all_reliability.append(edge_weight * scale_weight)
        all_streams.append(torch.ones(len(local_boxes), dtype=torch.long))

    if tiles:
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        reliability = torch.cat(all_reliability, dim=0)
        if args.save_dart_proposals:
            clusters = cluster_multiview_proposals(
                boxes,
                scores,
                reliability,
                torch.cat(all_streams, dim=0),
                iou_threshold=args.fusion_iou,
            )
            fused_boxes = clusters["boxes"]
            fused_scores = clusters["scores"]
            observations = clusters["observations"]
        else:
            fused_boxes, fused_scores, observations = size_aware_weighted_box_fusion(
                boxes,
                scores,
                reliability,
                iou_threshold=args.fusion_iou,
            )
    else:
        fused_boxes, fused_scores = full_boxes, full_scores
        observations = torch.ones(len(full_boxes), dtype=torch.long)
        if args.save_dart_proposals:
            clusters = cluster_multiview_proposals(
                full_boxes,
                full_scores,
                all_reliability[0],
                all_streams[0],
                iou_threshold=args.fusion_iou,
            )
    keep = fused_scores >= args.final_conf
    order = torch.argsort(fused_scores[keep], descending=True, stable=True)[: args.max_det]
    fused_boxes = fused_boxes[keep][order]
    fused_scores = fused_scores[keep][order]
    observations = observations[keep][order]
    if str(args.device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_gpu_memory_mib = (
        float(torch.cuda.max_memory_allocated()) / float(1024**2)
        if str(args.device) != "cpu" and torch.cuda.is_available()
        else None
    )
    record = {
        "image_path": str(image_path.resolve()),
        "width": image_width,
        "height": image_height,
        "tiles": len(tiles),
        "raw_global_detections": len(full_boxes),
        "raw_local_detections": int(sum(len(value) for value in all_boxes[1:])),
        "glaf_applied": bool(tiles),
        "density_threshold": args.density_threshold,
        "scene_gate": (
            {
                "confidence": args.scene_gate_confidence,
                "count_threshold": args.scene_gate_count,
                "max_median_reference_area": args.scene_gate_max_median_area,
                "confident_global_count": scene_gate_count,
                "median_reference_area": scene_gate_median_area,
            }
            if args.scene_gate_confidence is not None
            else None
        ),
        "elapsed_seconds": elapsed,
        "global_elapsed_seconds": global_elapsed,
        "global_peak_gpu_memory_mib": global_peak_gpu_memory_mib,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "predictions": serialize_predictions(fused_boxes, fused_scores, observations),
    }
    if args.save_dart_proposals:
        dart_order = torch.arange(len(clusters["boxes"]))
        dart_keep = clusters["scores"] >= args.final_conf
        dart_order = torch.argsort(
            clusters["scores"][dart_keep], descending=True, stable=True
        )[: args.max_det]
        selected = torch.nonzero(dart_keep, as_tuple=False).flatten()[dart_order]
        record["dart_proposals"] = [
            {
                "xyxy": [round(float(value), 3) for value in clusters["boxes"][index]],
                "confidence": round(float(clusters["scores"][index]), 8),
                "observations": int(clusters["observations"][index]),
                "global_observations": int(clusters["global_observations"][index]),
                "local_observations": int(clusters["local_observations"][index]),
                "max_confidence": round(float(clusters["max_score"][index]), 8),
                "global_max_confidence": round(float(clusters["global_max_score"][index]), 8),
                "local_max_confidence": round(float(clusters["local_max_score"][index]), 8),
                "mean_reliability": round(float(clusters["mean_reliability"][index]), 8),
                "score_std": round(float(clusters["score_std"][index]), 8),
                "mean_cluster_iou": round(float(clusters["mean_cluster_iou"][index]), 8),
                "box_dispersion": round(float(clusters["box_dispersion"][index]), 8),
            }
            for index in selected
        ]
    if args.save_global_predictions:
        global_keep = full_scores >= args.final_conf
        global_order = torch.argsort(
            full_scores[global_keep], descending=True, stable=True
        )[: args.max_det]
        record["global_predictions"] = serialize_predictions(
            full_boxes[global_keep][global_order],
            full_scores[global_keep][global_order],
        )
    return record


def main() -> None:
    args = parse_args()
    if args.density_threshold < 0:
        raise ValueError("density-threshold must be non-negative")
    scene_gate_values = (
        args.scene_gate_confidence,
        args.scene_gate_count,
        args.scene_gate_max_median_area,
    )
    if any(value is not None for value in scene_gate_values) and not all(
        value is not None for value in scene_gate_values
    ):
        raise ValueError("Set all three --scene-gate-* arguments together")
    if args.scene_gate_confidence is not None:
        if not 0.0 <= args.scene_gate_confidence <= 1.0:
            raise ValueError("scene-gate-confidence must be in [0, 1]")
        if args.scene_gate_count < 1:
            raise ValueError("scene-gate-count must be positive")
        if args.scene_gate_max_median_area <= 0:
            raise ValueError("scene-gate-max-median-area must be positive")
    if str(args.device) != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.device):
        raise RuntimeError(
            "Single-GPU inference requires CUDA_VISIBLE_DEVICES to equal --device"
        )
    images = resolve_images(args.source, args.source_split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        images = images[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    # CUDA_VISIBLE_DEVICES exposes one physical card, so it is logical cuda:0
    # inside this process even when the requested physical card is GPU1.
    args.device = "cpu" if str(args.device) == "cpu" else "0"
    with args.output.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(images, start=1):
            record = predict_one(model, image_path, args)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if not args.quiet:
                print(
                    f"[{index}/{len(images)}] {image_path.name}: "
                    f"detections={len(record['predictions'])} "
                    f"time={record['elapsed_seconds']:.3f}s",
                    flush=True,
                )
    print(f"predictions={args.output.resolve()}")


if __name__ == "__main__":
    main()

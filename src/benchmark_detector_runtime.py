"""Benchmark detector complexity and end-to-end inference on cached orchard images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import register_pruscope_modules
from src.evaluate_size_stratified import load_dataset

register_pruscope_modules()

from ultralytics import RTDETR, YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=WEIGHTS")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected NAME=WEIGHTS, got {specification}")
    name, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise FileNotFoundError(specification)
    return name, path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def main() -> None:
    args = parse_args()
    visible_devices = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if str(args.device) != "cpu" and (not visible_devices or "," in visible_devices):
        raise RuntimeError(
            "Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES and use --device 0"
        )
    if min(args.images, args.batch, args.repeats) < 1 or args.warmup < 0:
        raise ValueError("images, batch, and repeats must be positive; warmup cannot be negative")
    image_paths, _ = load_dataset(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    selected = sorted(
        rng.choice(image_paths, size=min(args.images, len(image_paths)), replace=False),
        key=str,
    )
    cached_images = [read_image(path) for path in selected]
    rows: list[dict[str, object]] = []
    raw_timings: dict[str, list[float]] = {}
    for model_name, weights in map(named_path, args.model):
        model_class = RTDETR if model_name.lower().startswith("rtdetr") else YOLO
        detector = model_class(str(weights))
        parameters = sum(parameter.numel() for parameter in detector.model.parameters())
        flops = float(get_flops(detector.model, imgsz=args.imgsz))
        common = {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "half": str(args.device) != "cpu",
            "verbose": False,
            "save": False,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 500,
        }
        # Warm the exact full-source execution path used for measurement.  The
        # previous single-batch warm-up left first-use preprocessing and
        # predictor setup in the first timed full-set pass.  That made the
        # reported mean depend on model order and produced incompatible A2
        # latency values in otherwise identical manuscript tables.
        for _ in range(args.warmup):
            detector.predict(source=cached_images, **common)
        if str(args.device) != "cpu":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        milliseconds_per_image: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            detector.predict(source=cached_images, **common)
            if str(args.device) != "cpu":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            milliseconds_per_image.append(elapsed * 1000.0 / len(cached_images))
        mean_ms = statistics.mean(milliseconds_per_image)
        row = {
            "model": model_name,
            "weights": str(weights.resolve()),
            "parameters": parameters,
            "gflops_at_imgsz": flops,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "images": len(cached_images),
            "repeats": args.repeats,
            "latency_ms_per_image_mean": mean_ms,
            "latency_ms_per_image_sd": statistics.stdev(milliseconds_per_image)
            if len(milliseconds_per_image) > 1
            else 0.0,
            "throughput_images_per_second": 1000.0 / mean_ms,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2)
            if str(args.device) != "cpu"
            else 0.0,
        }
        rows.append(row)
        raw_timings[model_name] = milliseconds_per_image
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del detector
        if str(args.device) != "cpu":
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runtime_benchmark.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "runtime_benchmark.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "split": args.split,
                "benchmark_protocol": {
                    "source": "cached decoded source-image arrays",
                    "timed_scope": "complete detector.predict call over the fixed image set",
                    "warmup_full_set_passes": args.warmup,
                    "timed_full_set_passes": args.repeats,
                    "cuda_synchronized_before_and_after_timing": str(args.device) != "cpu",
                    "batch": args.batch,
                    "imgsz": args.imgsz,
                    "confidence": 0.25,
                    "iou": 0.7,
                    "max_det": 500,
                },
                "selected_images": [str(path.resolve()) for path in selected],
                "results": rows,
                "raw_latency_ms_per_image": raw_timings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

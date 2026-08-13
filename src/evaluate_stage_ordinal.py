"""Evaluate a DCOH checkpoint with auditable predictions and bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import PruScopeROIStageModel
from src.train_stage_ordinal import STAGE_NAMES, stage_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-images",
        type=Path,
        help="Optional stage-image split CSV used to report ROI-availability coverage",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--image-condition",
        choices=("normal", "grayscale", "center_only", "background_only"),
        default="normal",
        help="Controlled visual intervention applied before normalization",
    )
    return parser.parse_args()


class CenterIntervention:
    """Keep or remove the centered fruit region in square ROI crops.

    Stage crops were generated with a 20%% margin around a centered detection
    box, so the central 72%% square is a reproducible foreground proxy. The
    complement is an approximate background-only view, not a segmentation.
    """

    def __init__(self, keep_center: bool, fraction: float = 0.72) -> None:
        self.keep_center = keep_center
        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).copy()
        height, width = array.shape[:2]
        region_height = max(1, int(round(height * self.fraction)))
        region_width = max(1, int(round(width * self.fraction)))
        y1 = (height - region_height) // 2
        x1 = (width - region_width) // 2
        center_mask = np.zeros((height, width), dtype=bool)
        center_mask[y1 : y1 + region_height, x1 : x1 + region_width] = True
        keep_mask = center_mask if self.keep_center else ~center_mask
        fill = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)
        array[~keep_mask] = fill
        return Image.fromarray(array)


def evaluation_transform(condition: str):
    operations = [transforms.Resize(232), transforms.CenterCrop(224)]
    if condition == "grayscale":
        operations.append(transforms.Grayscale(num_output_channels=3))
    elif condition == "center_only":
        operations.append(CenterIntervention(keep_center=True))
    elif condition == "background_only":
        operations.append(CenterIntervention(keep_center=False))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return transforms.Compose(operations)


class EvaluationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform, mean, std) -> None:
        self.rows = rows
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            crop = self.transform(image.convert("RGB"))
        geometry = torch.tensor(
            [float(row["log_width"]), float(row["log_height"]), float(row["log_area"])],
            dtype=torch.float32,
        )
        geometry = (geometry - self.mean) / self.std
        return crop, geometry, int(row["stage_index"]), index


def bootstrap_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if iterations < 1:
        return {}
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(targets == index) for index in range(len(STAGE_NAMES))]
    samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        metrics = stage_metrics(targets[indices], predictions[indices])
        for key, value in metrics.items():
            if key != "confusion_matrix":
                samples.setdefault(key, []).append(float(value))
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if str(args.device) != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 before GPU evaluation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if str(args.device) == "cpu" else "cuda:0")
    # Keep dataset-side normalization tensors on CPU. Loading the complete
    # checkpoint directly onto CUDA makes these tensors unsafe for spawned
    # Windows DataLoader workers (CUDA/CPU device mismatch in __getitem__).
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    geometry_mean = torch.as_tensor(
        checkpoint["geometry_mean"], dtype=torch.float32, device="cpu"
    )
    geometry_std = torch.as_tensor(
        checkpoint["geometry_std"], dtype=torch.float32, device="cpu"
    )
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    transform = evaluation_transform(args.image_condition)
    no_visual = bool(checkpoint.get("args", {}).get("no_visual", False))
    loader = DataLoader(
        EvaluationDataset(rows, transform, geometry_mean, geometry_std),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device) != "cpu",
    )
    model = PruScopeROIStageModel(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    probabilities = np.empty((len(rows), len(STAGE_NAMES)), dtype=np.float32)
    developmental_indices = np.empty(len(rows), dtype=np.float32)
    targets = np.empty(len(rows), dtype=np.int64)
    no_geometry = bool(checkpoint.get("args", {}).get("no_geometry", False))
    with torch.inference_mode():
        for crops, geometry, batch_targets, row_indices in loader:
            crops, geometry = crops.to(device), geometry.to(device)
            if no_geometry:
                geometry.zero_()
            outputs = model(crops, geometry, zero_visual=no_visual)
            indices = row_indices.numpy()
            probabilities[indices] = outputs["stage_probabilities"].cpu().numpy()
            developmental_indices[indices] = outputs["developmental_index"].cpu().numpy()
            targets[indices] = batch_targets.numpy()
    predictions = probabilities.argmax(axis=1)
    crop_rows: list[dict] = []
    for row, target, prediction, probability, index in zip(
        rows, targets, predictions, probabilities, developmental_indices, strict=True
    ):
        crop_rows.append(
            {
                "image_id": row["image_id"],
                "crop_path": row["crop_path"],
                "target": STAGE_NAMES[int(target)],
                "prediction": STAGE_NAMES[int(prediction)],
                **{
                    f"probability_{name}": float(value)
                    for name, value in zip(STAGE_NAMES, probability, strict=True)
                },
                "developmental_index": float(index),
            }
        )
    image_rows: list[dict] = []
    for image_id in sorted({row["image_id"] for row in rows}):
        selected = np.asarray([row["image_id"] == image_id for row in rows])
        mean_probability = probabilities[selected].mean(axis=0)
        target = int(targets[selected][0])
        prediction = int(mean_probability.argmax())
        image_rows.append(
            {
                "image_id": image_id,
                "crops": int(selected.sum()),
                "target": STAGE_NAMES[target],
                "prediction": STAGE_NAMES[prediction],
                **{
                    f"probability_{name}": float(value)
                    for name, value in zip(STAGE_NAMES, mean_probability, strict=True)
                },
                "mean_developmental_index": float(developmental_indices[selected].mean()),
            }
        )
    image_targets = np.asarray([STAGE_NAMES.index(row["target"]) for row in image_rows])
    image_predictions = np.asarray(
        [STAGE_NAMES.index(row["prediction"]) for row in image_rows]
    )
    coverage = None
    if args.expected_images is not None:
        with args.expected_images.open("r", encoding="utf-8-sig", newline="") as handle:
            expected_rows = [
                row for row in csv.DictReader(handle) if row["split"] == args.split
            ]
        expected_ids = {row["image_id"] for row in expected_rows}
        observed_ids = {row["image_id"] for row in image_rows}
        if not observed_ids <= expected_ids:
            raise RuntimeError("Crop manifest contains image IDs absent from --expected-images")
        per_stage = {}
        for stage in STAGE_NAMES:
            stage_expected = {row["image_id"] for row in expected_rows if row["stage"] == stage}
            stage_observed = stage_expected & observed_ids
            per_stage[stage] = {
                "expected": len(stage_expected),
                "with_crop": len(stage_observed),
                "without_crop": len(stage_expected - stage_observed),
                "fraction_with_crop": len(stage_observed) / len(stage_expected) if stage_expected else 0.0,
            }
        coverage = {
            "expected": len(expected_ids),
            "with_crop": len(observed_ids),
            "without_crop": len(expected_ids - observed_ids),
            "fraction_with_crop": len(observed_ids) / len(expected_ids) if expected_ids else 0.0,
            "missing_image_ids": sorted(expected_ids - observed_ids),
            "by_stage": per_stage,
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "no_geometry": no_geometry,
        "no_visual": no_visual,
        "image_condition": args.image_condition,
        "crop_metrics": stage_metrics(targets, predictions),
        "image_metrics": stage_metrics(image_targets, image_predictions),
        "image_bootstrap_95_ci": bootstrap_intervals(
            image_targets, image_predictions, args.bootstrap, args.seed
        ),
        "bootstrap_iterations": args.bootstrap,
        "image_coverage": coverage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "crop_predictions.csv", crop_rows)
    write_csv(args.output_dir / "image_predictions.csv", image_rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

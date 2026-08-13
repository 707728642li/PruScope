"""Train PruScope's Developmental Continuum Ordinal Head (DCOH)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import PruScopeROIStageModel
from src.train_detector_thermal import ThermalGovernor


STAGE_NAMES = ["small_green", "medium_green", "mature"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Ablation: replace normalized box geometry with zeros",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Ablation: bypass the visual encoder and use box geometry only",
    )
    parser.add_argument("--pause-temperature", type=int, default=83)
    parser.add_argument("--resume-temperature", type=int, default=80)
    return parser.parse_args()


class StageCropDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        transform,
        geometry_mean: torch.Tensor,
        geometry_std: torch.Tensor,
        load_visual: bool = True,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.geometry_mean = geometry_mean
        self.geometry_std = geometry_std
        self.load_visual = load_visual

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        if self.load_visual:
            with Image.open(row["crop_path"]) as image:
                crop = self.transform(image.convert("RGB"))
        else:
            # The geometry-only control bypasses the encoder, so avoid needless
            # image decoding and return a minimal placeholder tensor.
            crop = torch.zeros((3, 1, 1), dtype=torch.float32)
        geometry = torch.tensor(
            [float(row["log_width"]), float(row["log_height"]), float(row["log_area"])],
            dtype=torch.float32,
        )
        geometry = (geometry - self.geometry_mean) / self.geometry_std
        return crop, geometry, int(row["stage_index"]), row["image_id"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stage_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=(precision + recall) > 0,
    )
    total = confusion.sum()
    weights = (
        (np.arange(3)[:, None] - np.arange(3)[None, :]).astype(np.float64) ** 2
        / 4.0
    )
    observed = confusion / max(total, 1)
    expected = np.outer(support, predicted) / max(total * total, 1)
    expected_disagreement = float((weights * expected).sum())
    kappa = (
        1.0 - float((weights * observed).sum()) / expected_disagreement
        if expected_disagreement > 0
        else 1.0
    )
    result = {
        "accuracy": float(true_positive.sum() / max(total, 1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "ordinal_mae": float(np.abs(targets - predictions).mean()),
        "quadratic_weighted_kappa": kappa,
    }
    for index, name in enumerate(STAGE_NAMES):
        result[f"recall_{name}"] = float(recall[index])
        result[f"f1_{name}"] = float(f1[index])
    result["confusion_matrix"] = confusion.tolist()
    return result


@torch.inference_mode()
def evaluate(
    model: PruScopeROIStageModel,
    loader: DataLoader,
    device: torch.device,
    no_geometry: bool = False,
    no_visual: bool = False,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_image_ids: list[str] = []
    for crops, geometry, targets, image_ids in loader:
        crops, geometry, targets = (
            crops.to(device, non_blocking=True),
            geometry.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )
        if no_geometry:
            geometry.zero_()
        outputs = model(crops, geometry, zero_visual=no_visual)
        loss = model.ordinal_head.loss(outputs["cumulative_logits"], targets)
        predictions = outputs["stage_probabilities"].argmax(dim=1)
        losses.append(float(loss))
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(outputs["stage_probabilities"].cpu().numpy())
        all_image_ids.extend(image_ids)
    targets_array = np.concatenate(all_targets)
    probabilities_array = np.concatenate(all_probabilities)
    crop_metrics = stage_metrics(targets_array, np.concatenate(all_predictions))
    grouped_probabilities: dict[str, list[np.ndarray]] = {}
    grouped_targets: dict[str, int] = {}
    for image_id, target, probabilities in zip(
        all_image_ids, targets_array, probabilities_array, strict=True
    ):
        grouped_probabilities.setdefault(image_id, []).append(probabilities)
        grouped_targets[image_id] = int(target)
    ordered_ids = sorted(grouped_probabilities)
    image_targets = np.asarray([grouped_targets[value] for value in ordered_ids])
    image_predictions = np.asarray(
        [
            np.mean(grouped_probabilities[value], axis=0).argmax()
            for value in ordered_ids
        ]
    )
    image_metrics = stage_metrics(image_targets, image_predictions)
    image_metrics["images"] = len(ordered_ids)
    return float(np.mean(losses)), crop_metrics, image_metrics


def main() -> None:
    args = parse_args()
    if args.no_geometry and args.no_visual:
        raise ValueError("--no-geometry and --no-visual cannot be combined")
    use_cuda = args.device != "cpu"
    if use_cuda and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("DCOH GPU training requires CUDA_VISIBLE_DEVICES=0")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    if any(not split_rows[split] for split in split_rows):
        raise RuntimeError("train, val, and test splits must all be non-empty")
    train_geometry = torch.tensor(
        [
            [float(row["log_width"]), float(row["log_height"]), float(row["log_area"])]
            for row in split_rows["train"]
        ],
        dtype=torch.float32,
    )
    geometry_mean = train_geometry.mean(dim=0)
    geometry_std = train_geometry.std(dim=0).clamp_min(1e-6)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    datasets = {
        "train": StageCropDataset(
            split_rows["train"],
            train_transform,
            geometry_mean,
            geometry_std,
            load_visual=not args.no_visual,
        ),
        "val": StageCropDataset(
            split_rows["val"],
            eval_transform,
            geometry_mean,
            geometry_std,
            load_visual=not args.no_visual,
        ),
        "test": StageCropDataset(
            split_rows["test"],
            eval_transform,
            geometry_mean,
            geometry_std,
            load_visual=not args.no_visual,
        ),
    }
    class_counts = Counter(int(row["stage_index"]) for row in split_rows["train"])
    sample_weights = torch.tensor(
        [1.0 / class_counts[int(row["stage_index"])] for row in split_rows["train"]],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    common_loader = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.workers > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **common_loader),
        "val": DataLoader(datasets["val"], shuffle=False, **common_loader),
        "test": DataLoader(datasets["test"], shuffle=False, **common_loader),
    }
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = PruScopeROIStageModel(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.visual_encoder.parameters(), "lr": args.backbone_lr},
            {"params": model.ordinal_head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    governor = (
        ThermalGovernor(0, args.pause_temperature, args.resume_temperature, 8)
        if use_cuda
        else None
    )
    history: list[dict[str, float]] = []
    best_f1, stale_epochs = -1.0, 0
    best_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for crops, geometry, targets, _ in loaders["train"]:
            crops, geometry, targets = (
                crops.to(device, non_blocking=True),
                geometry.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            if args.no_geometry:
                geometry.zero_()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_cuda
            ):
                outputs = model(crops, geometry, zero_visual=args.no_visual)
                loss = model.ordinal_head.loss(outputs["cumulative_logits"], targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
            if governor is not None:
                governor.on_train_batch_end(None)
        val_loss, val_crop_metrics, val_image_metrics = evaluate(
            model, loaders["val"], device, args.no_geometry, args.no_visual
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            **{
                f"crop_{key}": value
                for key, value in val_crop_metrics.items()
                if key != "confusion_matrix"
            },
            **{
                f"image_{key}": value
                for key, value in val_image_metrics.items()
                if key != "confusion_matrix"
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_image_metrics["macro_f1"] > best_f1:
            best_f1 = val_image_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stage_names": STAGE_NAMES,
                    "geometry_mean": geometry_mean,
                    "geometry_std": geometry_std,
                    "epoch": epoch,
                    "val_crop_metrics": val_crop_metrics,
                    "val_image_metrics": val_image_metrics,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break
    if governor is not None:
        governor.on_train_end(None)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_crop_metrics, test_image_metrics = evaluate(
        model, loaders["test"], device, args.no_geometry, args.no_visual
    )
    report = {
        "best_epoch": checkpoint["epoch"],
        "validation_crop": checkpoint["val_crop_metrics"],
        "validation_image": checkpoint["val_image_metrics"],
        "test_loss": test_loss,
        "test_crop": test_crop_metrics,
        "test_image": test_image_metrics,
        "class_counts_train": dict(sorted(class_counts.items())),
        "geometry_mean": geometry_mean.tolist(),
        "geometry_std": geometry_std.tolist(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

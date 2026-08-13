"""Train the PruScope Density-Aware microfruit Refinement Tail (DART)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import DART_METADATA_NAMES, PruScopeDARTTail, encode_box_delta
from src.train_detector_thermal import ThermalGovernor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-initialize", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("0", "1", "cpu"), default="0")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-margin", type=float, default=0.35)
    parser.add_argument("--monitor-fraction", type=float, default=0.20)
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="Refit on every manifest row for the fixed --epochs after selection",
    )
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--box-weight", type=float, default=2.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--no-visual", action="store_true", help="Metadata-only DART ablation")
    parser.add_argument("--pause-temperature", type=int, default=83)
    parser.add_argument("--resume-temperature", type=int, default=80)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_task_ids(rows: list[dict], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministic group split stratified by stage and density tercile."""
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    groups: dict[tuple[str, int], list[str]] = {}
    positive_counts = {
        task: sum(int(row["target"]) for row in task_rows)
        for task, task_rows in by_task.items()
    }
    all_counts = np.asarray(list(positive_counts.values()), dtype=float)
    boundaries = np.quantile(all_counts, [1 / 3, 2 / 3]) if len(all_counts) else [0, 0]
    for task, task_rows in by_task.items():
        density = int(np.digitize(positive_counts[task], boundaries, right=True))
        groups.setdefault((task_rows[0]["stage"], density), []).append(task)
    monitor: set[str] = set()
    for key, task_ids in sorted(groups.items()):
        local_seed = seed + sum(ord(character) for character in f"{key[0]}-{key[1]}")
        rng = np.random.default_rng(local_seed)
        ordered = list(np.asarray(sorted(task_ids))[rng.permutation(len(task_ids))])
        count = max(1, round(len(ordered) * fraction)) if len(ordered) > 1 else 0
        monitor.update(ordered[:count])
    fit = set(by_task) - monitor
    if not fit or not monitor:
        raise RuntimeError("DART fit/monitor group split is empty")
    return fit, monitor


def expanded_square(box: list[float], width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    side = max(side, 8.0)
    left = max(0, min(width - 1, int(np.floor(center_x - side / 2))))
    top = max(0, min(height - 1, int(np.floor(center_y - side / 2))))
    right = max(left + 1, min(width, int(np.ceil(center_x + side / 2))))
    bottom = max(top + 1, min(height, int(np.ceil(center_y + side / 2))))
    return left, top, right, bottom


class DARTDataset(Dataset):
    def __init__(self, rows: list[dict], transform, metadata_mean, metadata_std, margin: float, load_visual: bool = True) -> None:
        self.rows = rows
        self.transform = transform
        self.metadata_mean = metadata_mean
        self.metadata_std = metadata_std
        self.margin = margin
        self.load_visual = load_visual
        self._cached_path: str | None = None
        self._cached_image: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, path: str) -> np.ndarray:
        if path != self._cached_path:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode {path}")
            self._cached_path, self._cached_image = path, image
        assert self._cached_image is not None
        return self._cached_image

    def __getitem__(self, index: int):
        row = self.rows[index]
        if not self.load_visual:
            crop_tensor = torch.zeros((3, 1, 1), dtype=torch.float32)
        elif row.get("crop_path"):
            crop = cv2.imdecode(
                np.fromfile(row["crop_path"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if crop is None:
                raise RuntimeError(f"Could not decode {row['crop_path']}")
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            image = self._image(row["image_path"])
            left, top, right, bottom = expanded_square(
                row["proposal_box"], row["width"], row["height"], self.margin
            )
            crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
        if self.load_visual:
            crop_tensor = self.transform(Image.fromarray(crop))
        metadata = torch.tensor(row["metadata"], dtype=torch.float32)
        metadata = (metadata - self.metadata_mean) / self.metadata_std
        proposal = torch.tensor(row["proposal_box"], dtype=torch.float32)
        target = int(row["target"])
        if target:
            delta = encode_box_delta(proposal, torch.tensor(row["gt_box"], dtype=torch.float32))
        else:
            delta = torch.zeros(4, dtype=torch.float32)
        return crop_tensor, metadata, float(target), delta, row["task_id"]


@torch.inference_mode()
def evaluate(model, loader, device, positive_weight, box_weight, uncertainty_weight, no_visual=False) -> dict[str, float]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    losses: list[float] = []
    box_errors: list[np.ndarray] = []
    for crops, metadata, targets, deltas, _ in loader:
        crops, metadata, targets, deltas = (
            crops.to(device, non_blocking=True), metadata.to(device, non_blocking=True),
            targets.to(device, non_blocking=True), deltas.to(device, non_blocking=True),
        )
        outputs = model(crops, metadata, zero_visual=no_visual)
        classification = nn.functional.binary_cross_entropy_with_logits(
            outputs["objectness_logit"], targets, pos_weight=positive_weight
        )
        positive = targets > 0.5
        if positive.any():
            residual = nn.functional.smooth_l1_loss(
                outputs["box_delta"][positive], deltas[positive], reduction="none"
            )
            variance = outputs["log_variance"][positive]
            uncertainty = (torch.exp(-variance) * residual + 0.5 * variance).mean()
            box = residual.mean()
            box_errors.append(residual.cpu().numpy())
        else:
            uncertainty = classification.new_zeros(())
            box = classification.new_zeros(())
        loss = classification + box_weight * box + uncertainty_weight * uncertainty
        losses.append(float(loss))
        all_targets.append(targets.cpu().numpy())
        all_scores.append(torch.sigmoid(outputs["objectness_logit"]).cpu().numpy())
    targets = np.concatenate(all_targets).astype(int)
    scores = np.concatenate(all_scores)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(targets[order])
    fp = np.cumsum(1 - targets[order])
    recall = tp / max(targets.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = np.zeros(101)
    for index, threshold in enumerate(np.linspace(0, 1, 101)):
        candidates = np.flatnonzero(recall >= threshold)
        if len(candidates):
            sampled[index] = envelope[candidates[0]]
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1)) if len(f1) else 0
    return {
        "loss": float(np.mean(losses)), "average_precision": float(sampled.mean()),
        "best_f1": float(f1[best]) if len(f1) else 0.0,
        "best_threshold": float(scores[order][best]) if len(f1) else 1.0,
        "box_smooth_l1": float(np.concatenate(box_errors).mean()) if box_errors else 0.0,
    }


def main() -> None:
    args = parse_args()
    use_cuda = args.device != "cpu"
    if use_cuda and os.environ.get("CUDA_VISIBLE_DEVICES") != args.device:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal --device")
    seed_everything(args.seed)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.refit_all:
        fit_ids = {row["task_id"] for row in rows}
        monitor_ids: set[str] = set()
        split_rows = {"fit": rows, "monitor": []}
    else:
        fit_ids, monitor_ids = split_task_ids(rows, args.monitor_fraction, args.seed)
        split_rows = {
            "fit": [row for row in rows if row["task_id"] in fit_ids],
            "monitor": [row for row in rows if row["task_id"] in monitor_ids],
        }
    fit_metadata = torch.tensor([row["metadata"] for row in split_rows["fit"]], dtype=torch.float32)
    metadata_mean = fit_metadata.mean(dim=0)
    metadata_std = fit_metadata.std(dim=0).clamp_min(1e-5)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, scale=(0.86, 1.0), ratio=(0.94, 1.06)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(4),
        transforms.ColorJitter(0.15, 0.15, 0.12, 0.02), transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(args.crop_size + 8), transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(), normalize,
    ])
    datasets = {"fit": DARTDataset(split_rows["fit"], train_transform, metadata_mean, metadata_std, args.crop_margin, load_visual=not args.no_visual)}
    if not args.refit_all:
        datasets["monitor"] = DARTDataset(split_rows["monitor"], eval_transform, metadata_mean, metadata_std, args.crop_margin, load_visual=not args.no_visual)
    class_counts = Counter(int(row["target"]) for row in split_rows["fit"])
    sample_weights = torch.tensor([
        1.0 / class_counts[int(row["target"])] for row in split_rows["fit"]
    ], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    common = {"batch_size": args.batch, "num_workers": args.workers, "pin_memory": use_cuda, "persistent_workers": args.workers > 0}
    loaders = {"fit": DataLoader(datasets["fit"], sampler=sampler, **common)}
    if not args.refit_all:
        loaders["monitor"] = DataLoader(datasets["monitor"], shuffle=False, **common)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = PruScopeDARTTail()
    initialization = {"source": None, "visual_tensors": 0}
    if args.visual_initialize and not args.no_visual:
        checkpoint = torch.load(args.visual_initialize, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        initialization = {
            "source": str(args.visual_initialize.resolve()),
            "visual_tensors": model.load_visual_encoder(state),
        }
    model.to(device)
    positive_weight = torch.tensor(
        args.negative_weight * class_counts.get(0, 1) / max(class_counts.get(1, 1), 1), device=device
    )
    optimizer = torch.optim.AdamW([
        {"params": model.visual_encoder.parameters(), "lr": args.backbone_lr},
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("visual_encoder.")], "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    governor = ThermalGovernor(int(args.device), args.pause_temperature, args.resume_temperature, 8) if use_cuda else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ap, stale = -1.0, 0
    history: list[dict] = []
    best_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for crops, metadata, targets, deltas, _ in loaders["fit"]:
            crops, metadata, targets, deltas = crops.to(device), metadata.to(device), targets.float().to(device), deltas.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                outputs = model(crops, metadata, zero_visual=args.no_visual)
                classification = nn.functional.binary_cross_entropy_with_logits(outputs["objectness_logit"], targets, pos_weight=positive_weight)
                positive = targets > 0.5
                residual = nn.functional.smooth_l1_loss(outputs["box_delta"][positive], deltas[positive], reduction="none")
                variance = outputs["log_variance"][positive]
                box = residual.mean()
                uncertainty = (torch.exp(-variance) * residual + 0.5 * variance).mean()
                loss = classification + args.box_weight * box + args.uncertainty_weight * uncertainty
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
            if governor: governor.on_train_batch_end(None)
        metrics = (
            evaluate(model, loaders["monitor"], device, positive_weight, args.box_weight, args.uncertainty_weight, args.no_visual)
            if not args.refit_all
            else {"loss": float("nan"), "average_precision": float("nan"), "best_f1": float("nan"), "best_threshold": float("nan"), "box_smooth_l1": float("nan")}
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"monitor_{key}": value for key, value in metrics.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        should_save = args.refit_all and epoch == args.epochs
        if should_save or (not args.refit_all and metrics["average_precision"] > best_ap):
            best_ap, stale = metrics["average_precision"], 0
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch,
                "metadata_names": DART_METADATA_NAMES, "metadata_mean": metadata_mean,
                "metadata_std": metadata_std, "initialization": initialization,
                "fit_task_ids": sorted(fit_ids), "monitor_task_ids": sorted(monitor_ids),
                "monitor_metrics": metrics, "args": vars(args),
            }, best_path)
        elif not args.refit_all:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break
    if governor: governor.on_train_end(None)
    report = {
        "mode": "fixed_epoch_all_data_refit" if args.refit_all else "group_disjoint_epoch_selection",
        "best_epoch": torch.load(best_path, map_location="cpu", weights_only=False)["epoch"],
        "best_monitor_AP": best_ap, "fit_images": len(fit_ids), "monitor_images": len(monitor_ids),
        "fit_rows": len(split_rows["fit"]), "monitor_rows": len(split_rows["monitor"]),
        "initialization": initialization, "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

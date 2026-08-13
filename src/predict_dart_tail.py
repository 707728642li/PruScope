"""Run the trained DART proposal tail and cache raw outputs once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import DART_METADATA_NAMES, PruScopeDARTTail, proposal_metadata_vector
from src.train_dart_tail import expanded_square


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("0", "1", "cpu"), default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-margin", type=float, default=0.35)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def read_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def main() -> None:
    args = parse_args()
    use_cuda = args.device != "cpu"
    if use_cuda and os.environ.get("CUDA_VISIBLE_DEVICES") != args.device:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal --device")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["metadata_names"]) != tuple(DART_METADATA_NAMES):
        raise ValueError("DART metadata schema does not match checkpoint")
    model = PruScopeDARTTail()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    metadata_mean = checkpoint["metadata_mean"].float().to(device)
    metadata_std = checkpoint["metadata_std"].float().to(device)
    transform = transforms.Compose([
        transforms.Resize(args.crop_size + 8), transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    rows = [json.loads(line) for line in args.proposals.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for ordinal, record in enumerate(rows, start=1):
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = read_image(record["image_path"])
            compute_started = time.perf_counter()
            proposals = record.get("dart_proposals", [])
            crops: list[torch.Tensor] = []
            metadata: list[list[float]] = []
            for proposal in proposals:
                left, top, right, bottom = expanded_square(
                    proposal["xyxy"], record["width"], record["height"], args.crop_margin
                )
                crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                crops.append(transform(Image.fromarray(crop)))
                metadata.append(
                    proposal_metadata_vector(
                        proposal, record["width"], record["height"], len(proposals)
                    )
                )
            outputs_by_mode: dict[str, dict[str, list]] = {
                mode: {"probability": [], "box_delta": [], "log_variance": []}
                for mode in ("full", "metadata_only")
            }
            for start in range(0, len(crops), args.batch):
                crop_batch = torch.stack(crops[start : start + args.batch]).to(device)
                metadata_batch = torch.tensor(
                    metadata[start : start + args.batch], dtype=torch.float32, device=device
                )
                metadata_batch = (metadata_batch - metadata_mean) / metadata_std
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda
                ):
                    mode_outputs = {
                        "full": model(crop_batch, metadata_batch),
                        "metadata_only": model(crop_batch, metadata_batch, zero_visual=True),
                    }
                for mode, outputs in mode_outputs.items():
                    outputs_by_mode[mode]["probability"].extend(
                        torch.sigmoid(outputs["objectness_logit"]).float().cpu().tolist()
                    )
                    outputs_by_mode[mode]["box_delta"].extend(
                        outputs["box_delta"].float().cpu().tolist()
                    )
                    outputs_by_mode[mode]["log_variance"].extend(
                        outputs["log_variance"].float().cpu().tolist()
                    )
            for index, proposal in enumerate(proposals):
                proposal["dart"] = {
                    mode: {
                        "probability": round(outputs_by_mode[mode]["probability"][index], 8),
                        "box_delta": [round(value, 8) for value in outputs_by_mode[mode]["box_delta"][index]],
                        "log_variance": [round(value, 8) for value in outputs_by_mode[mode]["log_variance"][index]],
                    }
                    for mode in outputs_by_mode
                }
            if use_cuda:
                torch.cuda.synchronize()
            record["dart_checkpoint"] = str(args.checkpoint.resolve())
            record["dart_tail_elapsed_seconds"] = time.perf_counter() - started
            record["dart_tail_compute_seconds"] = time.perf_counter() - compute_started
            record["dart_tail_peak_gpu_memory_mib"] = (
                float(torch.cuda.max_memory_allocated()) / float(1024**2)
                if use_cuda
                else None
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if not args.quiet:
                print(f"[{ordinal}/{len(rows)}] {Path(record['image_path']).name}: proposals={len(proposals)}", flush=True)
    print(f"predictions={args.output.resolve()}")


if __name__ == "__main__":
    main()

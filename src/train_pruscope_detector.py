"""Train PruScope detector ablations on one explicitly selected physical GPU."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruscope import activate_pruscope_detect_head, register_pruscope_modules

register_pruscope_modules()

from ultralytics import YOLO  # noqa: E402  (custom registration must happen first)
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

from src.train_detector_thermal import EpochStopper, ThermalGovernor  # noqa: E402


SEMANTIC_LAYER_MAPS = {
    "builtin_p2": {
        **{index: index for index in range(17)},
        23: 17,
        25: 19,
        26: 20,
        28: 22,
    },
    "p2": {**{index: index for index in range(23)}},
    "p2h": {**{index: index for index in range(23)}},
    "pruscope": {
        **{index: index for index in range(23)},
    },
}


def configure_reproducible_training(seed: int) -> None:
    """Seed model creation and replace Ultralytics' seed-invariant loader RNG.

    Ultralytics 8.4.115 seeds ``build_dataloader`` with a fixed constant, so
    changing ``train(seed=...)`` alone does not change the shuffled samples or
    their worker-side augmentations. The model must also be seeded before the
    custom P2/OSSA layers are instantiated. Keeping this compatibility shim in
    the project makes the correction explicit and avoids modifying site-packages.
    """
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    init_seeds(seed, deterministic=True)
    data_build = importlib.import_module("ultralytics.data.build")
    detect_train = importlib.import_module("ultralytics.models.yolo.detect.train")

    def seeded_build_dataloader(
        dataset,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        pin_memory: bool = True,
        device: torch.device | str = "cuda",
    ):
        dataset_len = len(dataset)
        batch = min(batch, dataset_len)
        sampler = (
            None
            if rank == -1
            else data_build.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else data_build.ContiguousDistributedSampler(dataset)
        )
        samples = len(sampler) if sampler is not None else dataset_len
        drop_last = drop_last and bool(batch) and dataset_len % batch != 0
        batches = (
            samples // batch if drop_last else math.ceil(samples / batch)
        ) if batch else 0
        device_type = getattr(device, "type", str(device).split(":")[0])
        nd = (
            data_build.get_torch_device_backend(device).device_count()
            if device_type not in {"cpu", "mps"}
            else 0
        )
        nw = min(
            os.cpu_count() // max(nd, 1),
            workers,
            0 if batches <= 1 else batches,
        )
        generator = torch.Generator()
        # rank=-1 denotes a single process; distributed ranks receive distinct
        # but reproducible streams without changing the requested base seed.
        generator.manual_seed(seed + max(rank, 0))
        pin_memory = nd > 0 and pin_memory
        pin_memory_device = (
            device_type
            if pin_memory
            and device_type in {"npu", "xpu"}
            and data_build.TORCH_1_13
            and not data_build.TORCH_2_7
            else None
        )
        return data_build.InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=data_build.seed_worker,
            generator=generator,
            drop_last=drop_last,
            **({"pin_memory_device": pin_memory_device} if pin_memory_device else {}),
        )

    # DetectionTrainer imported this name into its module namespace, so patch
    # both the source module and the already-bound trainer reference.
    data_build.build_dataloader = seeded_build_dataloader
    detect_train.build_dataloader = seeded_build_dataloader
    print(
        f"[seed] model_init={seed} dataloader={seed} deterministic=True",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["baseline", *sorted(SEMANTIC_LAYER_MAPS)],
        required=True,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=0.0007)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--box", type=float, default=9.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--device",
        choices=("0", "1"),
        default="0",
        help="Physical CUDA device exposed to this single-GPU training process.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--pause-temperature", type=int, default=83)
    parser.add_argument("--resume-temperature", type=int, default=80)
    return parser.parse_args()


def copy_semantic_layers(target: YOLO, source: YOLO, architecture: str) -> dict[str, int]:
    """Copy unchanged feature extractors despite index shifts introduced by P2/OSSA."""
    target_layers = target.model.model
    source_layers = source.model.model
    copied_tensors = 0
    copied_parameters = 0
    for target_index, source_index in SEMANTIC_LAYER_MAPS[architecture].items():
        target_layer = target_layers[target_index]
        source_layer = source_layers[source_index]
        target_state = target_layer.state_dict()
        source_state = source_layer.state_dict()
        if target_state.keys() != source_state.keys():
            raise RuntimeError(
                f"State keys differ for target {target_index} and source {source_index}"
            )
        mismatched = [
            key
            for key in target_state
            if target_state[key].shape != source_state[key].shape
        ]
        if mismatched:
            raise RuntimeError(
                f"Shape mismatch target {target_index} <- source {source_index}: "
                f"{mismatched[:3]}"
            )
        target_layer.load_state_dict(source_state, strict=True)
        copied_tensors += len(source_state)
        copied_parameters += sum(parameter.numel() for parameter in target_layer.parameters())
    if architecture in {"p2h", "pruscope"}:
        target_head = target_layers[-1]
        source_head = source_layers[-1]
        # PruScope scale 0 is the new P2 branch. Its P3-P5 branches (1-3)
        # deliberately preserve the source head widths and can be transferred
        # exactly from the three source branches (0-2).
        for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            target_branches = getattr(target_head, attribute)
            source_branches = getattr(source_head, attribute)
            for source_scale, target_scale in enumerate((1, 2, 3)):
                target_branches[target_scale].load_state_dict(
                    source_branches[source_scale].state_dict(), strict=True
                )
                state = source_branches[source_scale].state_dict()
                copied_tensors += len(state)
                copied_parameters += sum(
                    parameter.numel()
                    for parameter in target_branches[target_scale].parameters()
                )
    return {"tensors": copied_tensors, "parameters": copied_parameters}


def build_initialized_model(args: argparse.Namespace) -> YOLO:
    if args.resume:
        if args.initialize_from.name != "last.pt":
            raise ValueError("--resume requires --initialize-from to be a last.pt checkpoint")
        return YOLO(str(args.initialize_from))
    if args.architecture == "baseline":
        return YOLO(str(args.initialize_from))
    if args.config is None:
        raise ValueError("--config is required for non-baseline architectures")
    if args.architecture in {"p2h", "pruscope"}:
        activate_pruscope_detect_head()
    model = YOLO(str(args.config))
    source = YOLO(str(args.initialize_from))
    summary = copy_semantic_layers(model, source, args.architecture)
    total_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    print(
        f"[init] semantic tensors={summary['tensors']} "
        f"parameters={summary['parameters']}/{total_parameters} "
        f"({summary['parameters'] / total_parameters:.1%})",
        flush=True,
    )
    model.model.names = source.model.names
    # Mark the in-memory initialized network as pretrained so Ultralytics passes
    # it into the trainer's reconstructed model instead of discarding the
    # semantically transferred state.
    model.ckpt = {"epoch": -1}
    model.ckpt_path = str(args.initialize_from.resolve())
    return model


def main() -> None:
    args = parse_args()
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_device != args.device:
        raise RuntimeError(
            "Single-GPU isolation requires CUDA_VISIBLE_DEVICES to equal "
            f"--device ({args.device}); received {visible_device!r}"
        )
    configure_reproducible_training(args.seed)
    model = build_initialized_model(args)
    governor = ThermalGovernor(
        nvml_index=int(args.device),
        pause_temperature=args.pause_temperature,
        resume_temperature=args.resume_temperature,
        check_every_batches=2,
    )
    model.add_callback("on_train_batch_end", governor.on_train_batch_end)
    model.add_callback("on_train_end", governor.on_train_end)
    model.add_callback(
        "on_fit_epoch_end", EpochStopper(args.stop_after_epoch).on_fit_epoch_end
    )
    common = {
        # CUDA_VISIBLE_DEVICES exposes exactly one physical GPU, which is
        # therefore logical device 0 inside this process. This intentionally
        # avoids DDP/NCCL communication during independent dual-GPU stress runs.
        "device": "0",
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "plots": not args.no_plots,
    }
    if args.resume:
        model.train(resume=True, **common)
        return
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        val=not args.no_val,
        save=not args.no_save,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=2.0,
        box=args.box,
        close_mosaic=5,
        mosaic=args.mosaic,
        mixup=0.0,
        degrees=3.0,
        translate=0.10,
        scale=0.45,
        fliplr=0.5,
        seed=args.seed,
        deterministic=True,
        save_period=5,
        cache=False,
        amp=True,
        **common,
    )


if __name__ == "__main__":
    main()

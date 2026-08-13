"""Create publication-ready, provenance-aware PruScope dataset summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_size_stratified import image_to_label_path, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--plos-manifest", type=Path, required=True)
    parser.add_argument("--stage-splits", type=Path, required=True)
    parser.add_argument("--stage-crops", type=Path, required=True)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--plos-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-size", type=int, default=1024)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_counts(
    rows: list[dict[str, str]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        record: dict[str, object] = dict(zip(group_fields, key, strict=True))
        record["derived_images"] = len(values)
        if "source_image" in values[0]:
            record["unique_source_images"] = len(
                {str(Path(row["source_image"]).resolve()).lower() for row in values}
            )
        if "boxes" in values[0]:
            record["boxes"] = sum(int(row["boxes"]) for row in values)
        summaries.append(record)
    return summaries


def size_distribution(data_yaml: Path, split: str, reference_size: int) -> dict[str, int]:
    image_paths, _ = load_dataset(data_yaml, split)
    thresholds = (32**2, 96**2)
    counts: Counter[str] = Counter()
    for image_path in image_paths:
        label_path = image_to_label_path(image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            area = float(fields[3]) * float(fields[4]) * reference_size**2
            size = "small" if area < thresholds[0] else "medium" if area < thresholds[1] else "large"
            counts[size] += 1
    return {
        "images": len(image_paths),
        "all": sum(counts.values()),
        "small": counts["small"],
        "medium": counts["medium"],
        "large": counts["large"],
    }


def main() -> None:
    args = parse_args()
    if args.reference_size < 1:
        raise ValueError("--reference-size must be positive")
    detector_rows = read_csv(args.detector_manifest)
    plos_rows = read_csv(args.plos_manifest)
    stage_rows = read_csv(args.stage_splits)
    crop_rows = read_csv(args.stage_crops)

    detector_summary = grouped_counts(
        detector_rows, ("split", "stage", "annotation_type")
    )
    plos_summary = grouped_counts(plos_rows, ("split", "camera_model", "license"))

    crop_counts = Counter((row["split"], row["stage"]) for row in crop_rows)
    stage_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in stage_rows:
        key = (row["split"], row["stage"], row["annotation_type"])
        stage_groups.setdefault(key, []).append(row)
    stage_summary: list[dict[str, object]] = []
    for (split, stage, annotation_type), values in sorted(stage_groups.items()):
        stage_summary.append(
            {
                "split": split,
                "stage": stage,
                "annotation_type": annotation_type,
                "images": len(values),
                "unique_source_images": len(
                    {str(Path(row["source_image"]).resolve()).lower() for row in values}
                ),
                "roi_crops_after_filtering": crop_counts[(split, stage)],
            }
        )

    size_rows = []
    for domain, data_yaml in (
        ("internal_highres", args.internal_data),
        ("plos_external", args.plos_data),
    ):
        size_rows.append(
            {
                "domain": domain,
                "split": "test",
                "reference_size": args.reference_size,
                **size_distribution(data_yaml, "test", args.reference_size),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "detector_dataset_summary.csv", detector_summary)
    write_csv(args.output_dir / "external_plos_summary.csv", plos_summary)
    write_csv(args.output_dir / "stage_dataset_summary.csv", stage_summary)
    write_csv(args.output_dir / "test_object_size_summary.csv", size_rows)
    payload = {
        "area_definition": f"normalized bbox area * {args.reference_size}^2",
        "size_thresholds": {"small": "<32^2", "medium": "32^2 to <96^2", "large": ">=96^2"},
        "detector_dataset": detector_summary,
        "external_plos_dataset": plos_summary,
        "stage_dataset": stage_summary,
        "test_object_sizes": size_rows,
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

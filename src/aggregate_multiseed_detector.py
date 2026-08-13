"""Aggregate corrected-seed detector evaluations without pseudo-replication."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


MODEL_PATTERN = re.compile(r"^(a[0125])_s(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, domain: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for row in payload["results"]:
        match = MODEL_PATTERN.match(str(row["model"]))
        if not match or row["size"] not in {"all", "small"}:
            continue
        rows.append(
            {
                "domain": domain,
                "architecture": match.group(1),
                "seed": int(match.group(2)),
                "size": row["size"],
                "AP50": float(row["AP50"]),
                "AP50_95": float(row["AP50_95"]),
                "AR50": float(row["AR50"]),
                "AR50_95": float(row["AR50_95"]),
            }
        )
    return rows


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n_seeds": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.internal, "internal") + load_rows(args.external, "external_plum")
    summary_rows: list[dict] = []
    for domain in ("internal", "external_plum"):
        for architecture in ("a0", "a1", "a2", "a5"):
            for size in ("all", "small"):
                values = [
                    row["AP50_95"]
                    for row in rows
                    if row["domain"] == domain
                    and row["architecture"] == architecture
                    and row["size"] == size
                ]
                if len(values) != 3:
                    raise ValueError(
                        f"Expected three seeds for {domain}/{architecture}/{size}; got {len(values)}"
                    )
                summary_rows.append(
                    {
                        "domain": domain,
                        "architecture": architecture,
                        "size": size,
                        "metric": "AP50_95",
                        **summarize(values),
                    }
                )

    paired_rows: list[dict] = []
    row_lookup = {
        (row["domain"], row["architecture"], row["seed"], row["size"]): row
        for row in rows
    }
    contrasts = (
        ("a5", "a0"),
        ("a2", "a1"),
        ("a5", "a2"),
        ("a2", "a0"),
    )
    for domain in ("internal", "external_plum"):
        for size in ("all", "small"):
            for target, reference in contrasts:
                for seed in (20260805, 20260806, 20260807):
                    reference_value = row_lookup[(domain, reference, seed, size)]["AP50_95"]
                    target_value = row_lookup[(domain, target, seed, size)]["AP50_95"]
                    paired_rows.append(
                        {
                            "contrast": f"{target}_minus_{reference}",
                            "domain": domain,
                            "size": size,
                            "seed": seed,
                            "reference_architecture": reference,
                            "target_architecture": target,
                            "reference_AP50_95": reference_value,
                            "target_AP50_95": target_value,
                            "difference": target_value - reference_value,
                        }
                    )

    paired_summary = []
    for target, reference in contrasts:
        contrast = f"{target}_minus_{reference}"
        for domain in ("internal", "external_plum"):
            for size in ("all", "small"):
                values = [
                    row["difference"]
                    for row in paired_rows
                    if row["contrast"] == contrast
                    and row["domain"] == domain
                    and row["size"] == size
                ]
                paired_summary.append(
                    {
                        "contrast": contrast,
                        "domain": domain,
                        "size": size,
                        "metric": f"{target.upper()}_minus_{reference.upper()}_AP50_95",
                        **summarize(values),
                        "all_differences_positive": all(value > 0 for value in values),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_level_metrics.csv", rows)
    write_csv(args.output_dir / "architecture_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_seed_differences.csv", paired_rows)
    payload = {
        "seed_control": {
            "seeds": [20260805, 20260806, 20260807],
            "model_initialization_seeded_before_construction": True,
            "dataloader_generator_bound_to_experiment_seed": True,
            "summary_policy": "mean and sample SD across independent training seeds",
            "inference": "descriptive; n=3 is not used as a hypothesis test",
        },
        "seed_level_metrics": rows,
        "architecture_summary": summary_rows,
        "paired_seed_differences": paired_rows,
        "paired_difference_summary": paired_summary,
    }
    output = args.output_dir / "multiseed_detector_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

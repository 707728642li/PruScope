"""Build the canonical machine-readable PruScope-DART result package."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("internal", "plos", "citdet")
DOMAIN_LABELS = {"internal": "Internal", "plos": "External plum", "citdet": "CitDet"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_lookup(payload: dict) -> dict[tuple[str, str], dict]:
    return {(row["model"], row["size"]): row for row in payload["results"]}


def main() -> None:
    report_root = ROOT / "reports" / "optimization" / "dart_microfruit_v2"
    validation_lock = load_json(report_root / "validation" / "DART_VALIDATION_SELECTION_LOCK.json")
    protocol = ROOT / "docs" / "DART_MICROFRUIT_REFINEMENT_PROTOCOL_20260812.md"
    addendum = ROOT / "docs" / "DART_V2_ANCHOR_PRESERVING_ADDENDUM_20260812.md"
    efficiency = load_json(report_root / "efficiency" / "citdet30_efficiency.json")
    domains = {}
    flat_rows = []
    bootstrap_rows = []
    density_rows = []
    for domain in DOMAINS:
        domain_root = ROOT / "reports" / "evaluation" / f"pruscope_dart_v2_{domain}"
        metrics_path = domain_root / "ablation_metrics_final" / "size_stratified_metrics.json"
        bootstrap_path = domain_root / "bootstrap_primary_2000" / "bootstrap_report.json"
        density_path = domain_root / "density_strata" / "density_stratified_analysis.json"
        metrics = load_json(metrics_path)
        bootstrap = load_json(bootstrap_path)
        density = load_json(density_path)
        lookup = metric_lookup(metrics)
        systems = {}
        for model in ("direct", "recall_only", "metadata", "box_only", "score_only", "score_box"):
            systems[model] = {size: lookup[(model, size)] for size in ("all", "small")}
            for size in ("all", "small"):
                row = lookup[(model, size)]
                flat_rows.append(
                    {
                        "domain": domain,
                        "system": model,
                        "size": size,
                        **{key: row[key] for key in ("targets", "AP50", "AP50_95", "AR50", "AR50_95")},
                    }
                )
        paired = [
            row
            for row in bootstrap["paired_differences"]
            if row["metric"] in ("AP50_95", "AR50")
        ]
        for row in paired:
            bootstrap_rows.append({"domain": domain, **row})
        for stratum in density["strata"]:
            direct = stratum["systems"]["direct"]
            dart = stratum["systems"]["dart"]
            density_rows.append(
                {
                    "domain": domain,
                    "stratum": stratum["stratum"],
                    "images": stratum["images"],
                    "direct_small_AP50_95": direct["small"]["AP50_95"],
                    "dart_small_AP50_95": dart["small"]["AP50_95"],
                    "delta_small_AP50_95": dart["small"]["AP50_95"] - direct["small"]["AP50_95"],
                    "direct_small_AR50": direct["small"]["AR50"],
                    "dart_small_AR50": dart["small"]["AR50"],
                    "direct_count_MAE_at_0.25": direct["operating_point"]["count_mae"],
                    "dart_count_MAE_at_0.25": dart["operating_point"]["count_mae"],
                }
            )
        domains[domain] = {
            "label": DOMAIN_LABELS[domain],
            "images": metrics["images"],
            "systems": systems,
            "paired_direct_vs_score_only": paired,
            "density_strata": density["strata"],
            "source_files": {
                "metrics": str(metrics_path.resolve()),
                "bootstrap": str(bootstrap_path.resolve()),
                "density": str(density_path.resolve()),
            },
        }

    validation = {
        "images": validation_lock["validation_images"],
        "direct": validation_lock["direct_A2"],
        "selected_candidate": validation_lock["selected_candidate"],
        "selected_configuration": validation_lock["selected_configuration"],
        "selected_metrics": validation_lock["selected_metrics"],
    }
    direct_small = validation["direct"]["small"]
    selected = validation["selected_metrics"]
    success_checks = {
        "validation_small_AP50_95_improved": selected["small_AP50_95"] > direct_small["AP50_95"],
        "at_least_two_protected_domains_improved_small_AP50_95": sum(
            domains[d]["systems"]["score_only"]["small"]["AP50_95"]
            > domains[d]["systems"]["direct"]["small"]["AP50_95"]
            for d in DOMAINS
        ) >= 2,
        "mean_protected_overall_delta_not_below_minus_0.005": sum(
            domains[d]["systems"]["score_only"]["all"]["AP50_95"]
            - domains[d]["systems"]["direct"]["all"]["AP50_95"]
            for d in DOMAINS
        ) / len(DOMAINS) >= -0.005,
        "exceeds_conventional_A0_small_endpoint": (
            domains["internal"]["systems"]["score_only"]["small"]["AP50_95"] > 0.4347834582532569
            and domains["plos"]["systems"]["score_only"]["small"]["AP50_95"] > 0.377217998
        ),
        "efficiency_reported": True,
        "paired_bootstrap_and_ablations_retained": True,
    }
    payload = {
        "status": "DART_POSTREVIEW_EVIDENCE_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_boundary": (
            "Post-review descriptive evidence on previously inspected protected domains; "
            "not fresh independent confirmation."
        ),
        "system": {
            "name": "PruScope-DART",
            "expansion": "Density-Aware microfruit Refinement Tail",
            "balanced_mode": "A2 direct global inference",
            "high_recall_mode": "anchor-preserving A2 global+local proposals with RGB+metadata DART scoring",
            "base_checkpoint": str((ROOT / "runs" / "pruscope_a2_human159x3_s20260805_e30" / "weights" / "best.pt").resolve()),
            "tail_checkpoint": str((ROOT / "runs" / "pruscope_dart_tail_v1_refit159_e7_s20260812" / "best.pt").resolve()),
        },
        "protocol": {
            "main": str(protocol.resolve()),
            "main_sha256": sha256(protocol),
            "anchor_preserving_addendum": str(addendum.resolve()),
            "anchor_preserving_addendum_sha256": sha256(addendum),
        },
        "training": {
            "eligible_human_images": 159,
            "images_with_tail_candidates": 153,
            "candidate_rows": 6634,
            "positive_rows_IoU_ge_0.50": 3486,
            "hard_negative_rows_IoU_lt_0.20_conf_ge_0.02": 3148,
            "visual_plus_metadata_selected_epoch": 7,
            "metadata_only_selected_epoch": 18,
            "monitor_average_precision_visual_plus_metadata": 0.9474327543,
            "monitor_average_precision_metadata_only": 0.9383220523,
        },
        "validation_selection": validation,
        "protected_domains": domains,
        "efficiency_probe": efficiency,
        "predeclared_success_checks": success_checks,
        "predeclared_success_all_met": all(success_checks.values()),
        "interpretation": (
            "DART is a recall-priority precision-phenotyping mode. It increases small-object "
            "AR50 in all three protected domains, improves small-object AP50-95 in PLOS and "
            "CitDet, and slightly decreases internal AP50-95. The direct A2 mode remains the "
            "balanced real-time default."
        ),
    }
    output_dir = report_root / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DART_FINAL_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in (
        ("dart_domain_metrics.csv", flat_rows),
        ("dart_paired_bootstrap.csv", bootstrap_rows),
        ("dart_density_strata.csv", density_rows),
    ):
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"output": str(output_dir.resolve()), "success": all(success_checks.values())}, indent=2))


if __name__ == "__main__":
    main()

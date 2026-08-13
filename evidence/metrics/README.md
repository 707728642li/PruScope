# Aggregate metric sources

| File | Role |
|---|---|
| `competitive_detector_benchmark.json` | same-budget three-seed YOLO11m, RT-DETR-L, and PruScope A2 comparison |
| `human_audit_summary.csv` | frozen 240-image detector metrics by visual cohort |
| `count_summary.csv` | cross-stage visible-count estimates and bootstrap intervals |
| `stage_metrics_and_interventions.csv` | DCOH reference-region ablations and interventions |
| `stage_robustness.csv` | image-level intervention details |
| `fruit_bearing_stage_metrics.json` | strict detector-to-stage point estimates |
| `fruit_bearing_stage_bootstrap.json` | strict pipeline source-image bootstrap intervals |
| `citdet_summary.csv` | public CitDet direct/G0/MARS aggregate results |
| `dart_domain_metrics.csv` | direct and DART-system metrics by domain/size |
| `dart_paired_bootstrap.csv` | paired DART-minus-direct intervals |
| `dart_density_strata.csv` | density-tertile descriptive analysis |

These aggregate files do not replace the withheld per-image predictions required for a complete independent recomputation. Sample sizes, thresholds, and evidence roles must be retained when quoting results.

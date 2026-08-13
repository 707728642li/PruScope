# Manuscript evidence map

This map links the manuscript's principal claims to the frozen evaluation role and repository artifacts. Values are rounded as in the manuscript; machine-readable source tables retain greater precision.

| Claim | Evaluation role | Frozen value/effect | Repository evidence |
|---|---|---|---|
| CCPH restores capacity lost after naïve P2 insertion | three-seed controlled ablation | A2 exceeded A1 in all 12 seed × domain × size cells | `evidence/metrics/detector_ablation_multiseed.csv`; Figure 3 |
| A2 is competitive on small fruit | same-budget comparator experiment | internal/external small AP50–95 0.433/0.374; YOLO11m 0.412/0.360; RT-DETR-L 0.390/0.221 | `evidence/metrics/competitive_multiseed.csv`; Figure 6 |
| Small-green fruit remain limiting | protected 240-image human reference | small-green COCO-small AP50–95 0.205, AR50 0.340 | `evidence/metrics/human_audit_summary.csv`; Figures 2 and 5 |
| Public zero-tuning transfer is nontrivial | CitDet official test, no train/tune | COCO-small AP50 0.660, AP50–95 0.256, AR50 0.793 | `evidence/metrics/citdet_summary.csv`; Figure S2 |
| DART trades latency for recall | validation-frozen known-domain post-review analysis | small-object AR50 changes +0.0133/+0.0382/+0.0120; 20.1 vs 864.0 ms/image on efficiency probe | `evidence/metrics/dart_domain_metrics.csv`; Figure S4 |
| DCOH separates ordered cohorts | human reference regions | macro F1 0.9786; quadratic-weighted κ 0.9696 | `evidence/metrics/stage_summary.csv`; Figure 4 |
| Strict stage pipeline is localization-limited | detector-to-stage evaluation | fruit-bearing image macro F1 0.946; joint correctly staged fruit recall 0.459 | `evidence/metrics/stage_summary.csv`; Figure 6 |
| Visible counts transfer imperfectly | validation-selected operating point | MAE 3.60 internal and 8.11 external plum | `evidence/metrics/count_summary.csv`; Figure 5 |

## Evidence labels

- **Protected test:** no threshold selection or model choice on the evaluated labels.
- **Zero-tuning external:** no training or parameter selection on that external archive.
- **Confirmation:** one-time locked analysis after development on a separate subset.
- **Post-review descriptive:** known domains were available before method development; useful for mechanism and limitations, not independent confirmation.

## Non-claims

The manuscript does not establish longitudinal growth, physiological maturity, complete tree yield, cultivar-invariant performance, or autonomous harvesting safety. Apparent-size strata are defined on the evaluation canvas, and the stage index represents ordered visual cohorts.

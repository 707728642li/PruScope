# PruScope experiment report

## Post-review DART completion (2026-08-12)

The Density-Aware microfruit Refinement Tail was implemented, trained under a
prewritten protocol, frozen on v4 validation, and evaluated without subsequent
protected-domain tuning. It met all predeclared success checks but did not
produce a universal AP improvement. The robust effect was small-object recall:
paired AR50 differences were +0.0133 internally, +0.0382 on external plum, and
+0.0120 on CitDet, with all 95% source-image bootstrap intervals above zero.
Small AP50-95 changed by -0.0020, +0.0057, and +0.0004, respectively.

The v1 replacement policy was a documented negative experiment because all 48
finite-grid candidates violated the overall AP guardrail. Before protected DART
inference, a frozen v2 addendum retained exact global anchors and allowed only
local-only proposals. Required ablations showed that recall expansion alone
reduced AP, metadata alone was insufficient, RGB-plus-metadata scoring recovered
external ranking quality, and box offsets were negligible. DART therefore
becomes a high-recall offline mode; direct human-enhanced A2 remains the
balanced model.

The full report and machine-readable evidence are under
`reports/optimization/dart_microfruit_v2/final`. No external upload was
performed.

The manuscript overview has been synchronized accordingly: Figure 1c now shows
the separate human-enhanced A2+DART route, exact global-anchor preservation,
local-only scoring, disabled box-offset/uncertainty-penalty components, and the
balanced real-time versus high-recall offline operating modes. It does not draw
DART as an extension of the historical A5/GLAF lineage.

Status: working report, 2026-08-06. Values based on automatic pseudo-labels are
explicitly marked and must not be presented as manually verified ground truth.

## System definition

PruScope is one scale-stage coupled framework with two learned components:

1. a generic one-class fruit localizer with MPP, OSSA, CCPH, and adaptive GLAF;
2. a shared DCOH ordinal ROI head that assigns small-green, medium-green, or
   mature probabilities and a continuous developmental index in [0, 2].

This is a cross-stage phenotyping system, not longitudinal tracking of the same
fruit. The available data do not support claims of individual-fruit growth
trajectories through time.

## Strict detector A3 result

Checkpoint: `runs/pruscope_m_strict_v1_gpu0/weights/best.pt` (best epoch 9).
The frozen 120-image small-green holdout was excluded from initialization and
training. This detector is valid for the reported localization experiments but
not for a final stage end-to-end experiment because mature stage validation/test
images were still present as generic localization pseudo-labels.

| Test domain | Mode | All AP50 | All AP50-95 | Small AP50 | Small AP50-95 | Small AR50 |
|---|---:|---:|---:|---:|---:|---:|
| PLOS external plum | baseline YOLO26m | 0.7832 | 0.5527 | 0.6007 | 0.3696 | 0.8913 |
| PLOS external plum | PruScope direct | 0.7845 | 0.5727 | 0.6027 | 0.3837 | 0.8996 |
| PLOS external plum | PruScope adaptive GLAF | 0.8015 | 0.5954 | 0.6509 | 0.4380 | 0.9403 |
| Internal high-resolution | baseline YOLO26m | — | — | 0.6436 | 0.4057 | — |
| Internal high-resolution | PruScope direct | 0.8423 | 0.6077 | 0.6743 | 0.4342 | 0.9600 |
| Internal high-resolution | PruScope adaptive GLAF | 0.8283 | 0.6010 | 0.6490 | 0.4222 | 0.9429 |

On the external PLOS test, adaptive GLAF improves small-fruit AP50-95 by 6.84
points over the baseline and by 5.43 points over direct PruScope. On the internal
high-resolution test, direct PruScope is preferable; this confirms that GLAF is
an adaptive dense-microfruit mode rather than a universal replacement for global
inference.

## Adaptive GLAF selection

The density gate uses the count of global detections at confidence 0.005. GLAF
is enabled when this count is at least 60. The threshold was selected using only
the internal high-resolution validation set and PLOS validation set, then frozen.

| Threshold | PLOS val all AP50-95 | PLOS val small AP50-95 | Internal val all AP50-95 | Internal val small AP50-95 | Four-metric mean |
|---:|---:|---:|---:|---:|---:|
| 0 (always GLAF) | 0.5173 | 0.2556 | 0.5938 | 0.3286 | 0.4238 |
| **60 (selected)** | **0.5214** | **0.2498** | **0.6070** | **0.3199** | **0.4245** |
| 80 | 0.5209 | 0.2495 | 0.6073 | 0.3183 | 0.4240 |
| 100 | 0.5164 | 0.2448 | 0.6077 | 0.3162 | 0.4213 |
| 120 | 0.5161 | 0.2450 | 0.6080 | 0.3185 | 0.4219 |
| 10000 (global only) | 0.4931 | 0.2017 | 0.6083 | 0.3188 | 0.4055 |

## DCOH stage result

Training uses detector-consistency filtering only on mature training crops.
Validation and test crops are not filtered. Capture groups are disjoint across
train, validation, and test. The small-green and mature boxes remain automatic;
the medium-green boxes are human annotations.

| Model | Crop macro F1 | Crop QWK | Image macro F1 | Image accuracy | Image QWK | Mature image recall |
|---|---:|---:|---:|---:|---:|---:|
| Visual-only ordinal head | 0.9312 | 0.9285 | 0.9609 | 0.9774 | 0.9487 | 0.8667 |
| **DCOH visual + geometry** | **0.9778** | **0.9696** | **0.9726** | **0.9812** | **0.9658** | **0.9000** |

The geometry coupling improves independent-test image macro F1 by 1.17 points,
image QWK by 1.71 points, and mature-image recall by 3.33 points. At crop level,
macro F1 improves by 4.66 points.

Stratified image-level bootstrap with 2,000 replicates gives DCOH macro F1
95% CI 0.9432--0.9943 and QWK 95% CI 0.9214--0.9958. The visual-only intervals
are 0.9259--0.9873 and 0.8920--0.9915, respectively. In a paired bootstrap on
the same images, the macro-F1 improvement is +0.0117 (95% CI -0.0084--0.0369)
and the QWK improvement is +0.0171 (-0.0084--0.0533). The point estimates
consistently favor geometry, but the paired intervals include zero; this should
be reported as an encouraging ablation rather than definitive significance.

Main checkpoint: `runs/pruscope_dcoh_v1_gpu0/best.pt` (best epoch 19).
Visual-only ablation: `runs/pruscope_dcoh_nogeom_v1_gpu0/best.pt` (best epoch 9).

## Final A5 strict detector

Dataset: `work/datasets/unified_fruit_v4_stage_strict`.

- 194 protected stage IDs are absent from detector training.
- train/validation/test source-image overlap is zero.
- training contains 1,638 small-green, 1,694 repeated medium-green, 159 mature,
  and 193 external PLOS images.
- initialization is from the medium-only teacher, not from A3.

Training run: `runs/pruscope_m_a5_stage_strict_gpu0`. Training completed for 10
epochs without thermal pauses. The frozen checkpoint is `weights/best.pt` from
epoch 9 (validation AP50 0.8007, AP50-95 0.5858); the last epoch was not selected.

| Independent test domain | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| Internal high-resolution | 0.7940 | 0.7576 | 0.8411 | 0.6115 |
| PLOS external plum | 0.7726 | 0.7067 | 0.7840 | 0.5722 |

Relative to A3, A5 improves AP50-95 by 0.37 points internally and 0.27 points
externally despite enforcing the stricter stage-image exclusion. This checkpoint
is therefore the final leakage-controlled detector for the end-to-end experiment.

The following table uses the same custom COCO-style size evaluator for both
direct and adaptive-GLAF predictions. Its all-size values differ slightly from
Ultralytics' standard evaluator above because the implementations are different.

| Test domain | Mode | All AP50 | All AP50-95 | Small AP50 | Small AP50-95 | Small AR50 |
|---|---:|---:|---:|---:|---:|---:|
| Internal high-resolution | A5 direct | 0.8406 | 0.6121 | 0.6772 | 0.4434 | 0.9600 |
| Internal high-resolution | A5 adaptive GLAF | 0.8214 | 0.6023 | 0.6517 | 0.4292 | 0.9467 |
| PLOS external plum | A5 direct | 0.7849 | 0.5739 | 0.5833 | 0.3758 | 0.8937 |
| PLOS external plum | **A5 adaptive GLAF** | **0.8033** | **0.5973** | **0.6447** | **0.4382** | **0.9498** |

On PLOS, the frozen gate improves small AP50-95 by 6.24 points and small AR50
by 5.61 points over A5 direct inference. Direct inference remains preferable on
the internal high-resolution set, so deployment should retain both routes and
activate local tiling only through the validation-selected density gate.

## Historical automatic-label detector plus DCOH result

The leakage-controlled A5 detector, adaptive GLAF threshold 60, detection
confidence 0.25, and DCOH were evaluated in series at IoU 0.50. Missed fruit
are counted as end-to-end failures rather than silently removed.

| Quantity | Result |
|---|---:|
| Test images | 277 |
| Ground-truth fruit | 2,892 |
| Predicted fruit | 3,061 |
| IoU-matched fruit | 2,235 |
| Detection recall at IoU 0.50 | 0.7728 |
| Stage accuracy on matched fruit | 0.9964 |
| Joint correct-stage recall over all ground truth | 0.7701 |
| Count MAE per image | 3.2310 |

| Target stage | Correctly detected and staged | Wrong stage after match | Missed detection |
|---|---:|---:|---:|
| Small green | 638 | 1 | 293 |
| Medium green | 1,191 | 3 | 229 |
| Mature | 398 | 4 | 135 |

This 277-image result is retained as a historical automatic-label engineering
analysis only. It is superseded for absolute reporting by the human-verified
cross-stage audit below.

## Final 240-image human-verified cross-stage audit

Two annotators completed disjoint assigned ranges (T0001-T0160 and T0161-T0240).
All 240 images were full-image confirmed. The immutable truth contains 5,151
fruit boxes: 1,965 small-green, 2,253 medium-green, and 933 mature. One
near-identical model/model duplicate was removed by the documented merge rule;
the final YOLO, COCO, and internal JSON representations are mutually consistent.

The frozen A5/GLAF detector achieved AP50 0.622, AP50-95 0.498, and AR50
0.708. At confidence 0.25 and IoU 0.50, precision/recall/F1 were
0.944/0.461/0.619. Count MAE was 11.15 fruit/image and bias was -10.99,
showing that dense-scene false negatives dominate. Small-green COCO-small
AP50-95 was 0.205 and AR50 was 0.340.

On 232 images with at least one human fruit ROI, DCOH image-level accuracy was
0.983, macro F1 was 0.979, and quadratic-weighted kappa was 0.970. The complete
bootstrap intervals, ablations, and interventions are in
`reports/evaluation/pruscope_human_240_final/MANUSCRIPT_READY_RESULTS.md`.
These results were computed without retraining. The now-inspected 240-image set
must not be used for iterative optimization.

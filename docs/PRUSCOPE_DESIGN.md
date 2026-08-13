# PruScope method design

## Working model name

**PruScope** — a scale–stage coupled orchard phenotyping network for *Prunus* fruit.

Proposed English manuscript title:

> **PruScope: A Unified Scale–Stage Coupled Framework for Cross-Stage Phenotyping of Apricot–Plum Fruit from Microfruit to Maturity in Unstructured Orchards**

Proposed Chinese title:

> **PruScope：面向非结构化果园李子果实从微幼果到成熟期跨阶段表型监测的尺度–阶段耦合统一框架**

“Whole-development-cycle” is used instead of “longitudinal dynamic tracking” because
the available images cover three developmental cohorts but do not repeatedly observe
the same fruit through time.

## Scientific task

PruScope will produce four phenotypes from ordinary RGB orchard images:

1. fruit location;
2. fruit count and count error per image;
3. apparent fruit-size distribution from normalized bounding-box area;
4. ordered developmental state: small green, medium/large green, and mature.

The detector remains a single generic `fruit` class so that human PLOS labels and all
three in-house stages can train the same localization representation. A developmental
ordinal head is applied to detected fruit features rather than training three isolated
detectors. This preserves cross-stage transfer and permits a continuous developmental
index in addition to three named stages.

## Architecture contributions

### 1. Microfruit-preserving pyramid

A non-invasive P2/4 prediction branch is appended to the already trained P3–P5 PAN.
The P2 branch retains shallow texture and boundary information that is otherwise lost
when immature fruit occupies only a few pixels, while the original medium/large-fruit
path remains structurally and numerically transferable from the best baseline.

### 2. Orchard Scale-Selective Attention (OSSA)

OSSA uses depthwise branches with effective 3×3, 5×5, and 7×7 receptive fields.
Per-channel softmax weights select the useful context scale, and a spatial gate
suppresses leaves, branches, highlights, and other orchard clutter. OSSA is inserted
only at P2/P3 levels to concentrate extra computation on small and medium fruit.
Residual layer scale is initialized at zero for stable transfer from pretrained YOLO.

### 3. Developmental Continuum Ordinal Head (DCOH)

The stages have a biological order rather than three unrelated identities. The head
will therefore learn two cumulative transition probabilities instead of an ordinary
three-way softmax. Training uses crop/ROI features and masks the stage loss for PLOS
boxes whose developmental stage is not supplied. The output supports both discrete
stages and a continuous 0–1 developmental index.

### 4. Cross-scale capacity-preserving detection head

Standard YOLO detection heads derive all branch widths from the first pyramid input.
Prepending a narrow P2 feature therefore reduces the capacity of P3–P5 heads as an
unintended side effect. PruScope derives the shared hidden widths from P3 instead.
This adds microfruit capacity without weakening medium-green and mature-fruit heads,
and permits exact transfer of the existing P3–P5 detection weights.

### 5. Global–local adaptive fusion

Full-image inference preserves context and large fruit. Overlapping high-resolution
tiles recover microfruit. Size-aware weighted box fusion will merge both streams while
preventing duplicate counts in tile overlaps.

## Required ablations

| ID | Detector | P2 | capacity-preserving head | OSSA | global–local fusion | ordinal stage head |
|---|---|---|---|---|---|---|
| A0 | current YOLO26m student | no | no | no | no | no |
| A1 | YOLO26m-P2 | yes | no | no | no | no |
| A2 | PruScope-P2H | yes | yes | no | no | no |
| A3 | PruScope-Det | yes | yes | yes | no | no |
| A4 | PruScope-Det-GL | yes | yes | yes | yes | no |
| A5 | full PruScope | yes | yes | yes | yes | yes |

## Leakage-control protocol

- All detector models reported on the frozen 120-image microfruit cohort must be
  initialized from the medium-green-only teacher, never from a student that has
  previously consumed those pseudo labels.
- Camera/capture-source groups are used only to keep correlated images on one
  side of a split; they are neither prediction targets nor model inputs.
- The stage branch uses unique source images (not the repeated detector
  hardlinks). Current image-level stage splits are: small green 1471/167/120,
  medium green 847/119/118, and mature 159/35/39 for train/validation/test.
- The 120 small-green test images retain the status
  `auto_consensus_review_candidate` until a compact human correction pass. They
  may support provisional engineering comparisons but not final manuscript
  ground-truth claims before that correction.

Primary detection endpoints are mAP50, mAP50–95, COCO-equivalent AP-small/medium/large,
AR-small, count MAE/RMSE, parameters, FLOPs, and latency. Stage endpoints are macro-F1,
balanced accuracy, quadratic-weighted kappa, ordinal MAE, and confusion matrices.

## Journal positioning

The contribution is framed as high-throughput, label-efficient orchard phenotyping,
not merely an “improved YOLO.” It combines microfruit sensing, cross-stage transfer,
ordered developmental inference, fruit-load quantification, and cross-camera external
validation. This aligns with Plant Phenomics work on small-fruit feature pyramids,
domain-adaptive auto-labeling, multiscale phenotyping, and unified crop counting.

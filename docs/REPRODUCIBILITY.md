# Reproducibility guide

## Evidence layers

PruScope separates four evidence layers:

1. **Architecture tests:** controlled multi-seed comparisons of A0/A1/A2/A5.
2. **Protected human reference:** complete cross-stage localization, counting, and stage evaluation.
3. **Public zero-tuning external evidence:** CitDet official test partition without training or tuning.
4. **Post-review development:** MARS and DART, explicitly labelled by validation/development/confirmation status.

Do not promote a post-review known-domain analysis to independent confirmation.

## Frozen units and controls

- Source image is the bootstrap and split unit.
- Derived images/crops stay with their source group.
- Detector thresholds, GLAF gate, count threshold, and DART validation grid are frozen before the paired test replay they govern.
- Three-seed results report the complete corrected seed set, not a best seed.
- Cached paired bootstraps compare predictions on the same source images.
- No-detection outcomes remain in strict detector-to-stage evaluation.

## Re-running aggregate audits

Use the synthetic tutorial to verify the repository's calculation/checking layer. Aggregate frozen results can be inspected in `evidence/metrics/`; main-figure source tables are copied alongside the scripts where available.

## Re-running training and inference

A complete neural-model reproduction requires assets not distributed in this candidate:

- approved internal split manifests and images;
- source annotations;
- compatible pretrained starting weights;
- frozen detector/DCOH/DART checkpoints;
- an environment lock generated without local paths;
- documented checkpoint SHA-256 values.

Once those rights are approved, the stable release should add a weight manifest, a small redistributable image fixture, CPU/GPU load smoke tests, a clean-machine run log, and a versioned archival DOI.

## Figure regeneration

Final figure scripts are in `paper/scripts/`. Some qualitative panels depend on restricted orchard images and per-image predictions and cannot be regenerated from this lightweight repository. Quantitative figure regeneration requires the frozen result tables documented in the evidence map. The committed PNG/PDF/SVG outputs are the review artifacts and are covered by `release/SHA256SUMS`.

## Determinism boundary

Seeds and grouped inputs are fixed, but GPU kernels, dependency versions, data-loader scheduling, and image shapes can produce small numerical differences. Report both the requested seed and the actual framework/runtime versions. Do not treat bitwise identity as a substitute for source-group integrity and paired statistical evaluation.

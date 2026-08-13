# User guide

## Choose an operating mode

- **Direct A2:** balanced mode for routine evaluation and the default starting point.
- **GLAF:** validation-gated local inference for scale/density-selected scenes.
- **DART:** offline recall-critical refinement; substantially slower and not suitable for real-time claims.
- **DCOH:** ordered stage assignment on detected or reference fruit crops.

Do not compare metrics across these modes without retaining their checkpoint, canvas size, confidence threshold, IoU rule, gate, and evidence status.

## Detector workflow

1. Prepare an approved local dataset in YOLO detection format.
2. Create an untracked dataset YAML with source-group-disjoint `train`, `val`, and `test` roots.
3. Select a model architecture from `configs/models/`.
4. Train with `src/train_pruscope_detector.py`, specifying the seed before construction.
5. Freeze the selected checkpoint and all validation-selected thresholds.
6. Materialize predictions with `src/predict_pruscope_global_local.py` or the DART scripts.
7. Evaluate apparent-size strata and paired differences with the provided evaluation scripts.

## Stage workflow

1. Build fruit crops from approved boxes while retaining the source-image group.
2. Generate group-disjoint cohort splits.
3. Train DCOH with `src/train_stage_ordinal.py`.
4. Evaluate reference-region stage discrimination with `src/evaluate_stage_ordinal.py`.
5. Evaluate the strict detector-to-stage pipeline with `src/evaluate_end_to_end_stage.py`; no-detection images must remain in the denominator.

## Inputs

Dataset paths in committed YAMLs are templates. A local data configuration must define image roots and one `fruit` class. Image identifiers must not be used to split derived crops independently of their source group.

## Outputs

Training scripts produce framework checkpoints, logs, and validation summaries. Prediction scripts produce JSON Lines records with image identifiers, original dimensions, runtime, and box lists. Evaluation scripts produce JSON/CSV metric summaries. Treat all per-image records as potentially sensitive until filenames and metadata are reviewed.

## Interpretation rules

- A predicted box describes a visible fruit hypothesis, not a unique fruit tracked through time.
- Count is visible count for one view, not total tree yield.
- Small/medium/large are canvas-relative apparent sizes.
- DCOH index encodes ordered visual cohorts, not days after flowering or physiological maturity.
- Low-confidence or absent predictions in dense canopies should trigger human review rather than silent acceptance.

## Versioning and migration

The current candidate is `1.0.0-rc.1`. No backward-compatibility promise applies before the stable release. Future stable breaking changes will require a major version; metric-definition or default changes will be documented in `CHANGELOG.md` and the evidence map.

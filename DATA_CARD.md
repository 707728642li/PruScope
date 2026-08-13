# PruScope data card

## Purpose

The PruScope evidence base supports fruit localization, visible counting, apparent-size stratification, and ordered visual-stage phenotyping in single-view orchard RGB images. The data are cross-sectional, not repeated observations of identified fruit.

## Data components

| Component | Role | Public files included here | Availability/terms |
|---|---|---:|---|
| Internal plum imagery | training, validation, protected testing, human audit | No | available from the manuscript correspondent only if institutional review permits |
| Internal bounding boxes | localization reference | No | not redistributed in this candidate |
| External plum archive | external localization evidence | No | original Figshare source; reported CC BY 4.0 |
| CitDet official test partition | zero-tuning external localization evidence | No | original source; reported CC BY-NC-SA 4.0 |
| Synthetic metric fixture | CI and reviewer audit example | Yes | original synthetic tabular data in this repository; covered by the repository's current no-license notice |
| Aggregate metrics and protocol locks | manuscript audit | Yes | no raw image pixels or bounding-box coordinates |

## Cohorts and labels

The internal data contain three operational visual cohorts: `small_green`, `medium_green`, and `visually_mature`. The detector uses one class, `fruit`. DCOH maps fruit crops to ordered cohort probabilities and an index from 0 to 2. Labels do not encode cultivar, chronological age, physiological maturity, firmness, sugar content, or harvest safety.

## Annotation and quality control

Human reviewers completed full-image fruit-box correction, including faint, defocused, occluded, truncated, and duplicate-appearance targets. The protected 240-image audit reference contains 5,151 verified fruit. Model-assisted preparation is not treated as ground-truth provenance by itself; reported complete-reference localization metrics rely on exhaustive human completion.

## Splits, grouping, and leakage controls

- Source image is the experimental unit for localization and counting.
- Derived resolutions and repeated training entries remain within the same source group.
- Training, validation, and protected test groups are disjoint under the recorded capture/source identifiers.
- Thresholds and scene gates are selected on validation data before their corresponding test evaluations.
- CitDet official test data contributed no training examples or parameter choices in the zero-tuning experiment.
- DART development domains were already known and are labelled post-review descriptive evidence rather than fresh confirmation.

The public-lightweight repository provides aggregate split/evidence locks but withholds filenames and annotation coordinates to avoid disclosing restricted data.

## Known biases

- Stage and acquisition conditions are correlated because different operators and sessions captured the cohorts.
- Dense microfruit, deep shadow, defocus, heavy occlusion, and boundary truncation are underdetected.
- The mature cohort represents external color development only.
- Camera distance and image resolution affect apparent object size.
- External validation covers a limited set of archives and cannot establish universal orchard generalization.

## Privacy and ethics

The intended subject is fruit, but orchard images can incidentally encode people, land, equipment, timestamps, or geolocation. Those risks must be reviewed before any image release. No annotator names, login identifiers, image metadata, internal filenames, or raw orchard photographs are included here.

## Redistribution and withdrawal

No internal data redistribution permission is asserted. External archives must be obtained from the original providers under their terms. If a source record is corrected or withdrawn, users should stop redistribution, record the affected version/checksum, and contact the repository maintainer.

## Citation

Users must cite both PruScope and every upstream dataset actually used. Dataset DOIs and notices are listed in `THIRD_PARTY_NOTICES.md`.

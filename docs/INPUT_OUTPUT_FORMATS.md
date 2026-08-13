# Input and output formats

## Detector dataset input

Model scripts expect a YOLO-style dataset configured by YAML. Image labels use one text row per object:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`; `class_id` is `0` for `fruit`. Empty images use an empty label file. Every derived image must retain a source-group identifier outside the label file for leakage-safe splitting.

## Prediction JSON Lines

Prediction writers emit one JSON object per source image. Common fields include an image identifier, original width/height, runtime, inference mode, and a box list. Each box contains `xyxy`, confidence, and class. Exact optional fields depend on the script and are visible in `--help` and source.

Per-image outputs can reveal internal filenames and are excluded from this repository.

## Aggregate metric tables

Tables in `evidence/metrics/` use explicit domain, method, size stratum, sample size, estimate, confidence interval, and evidence-role columns where applicable. Missing intervals are blank rather than encoded as zero.

## Synthetic audit CSV

Required columns:

| Column | Type | Constraint |
|---|---|---|
| `image_id` | string | non-empty and unique |
| `reference_count` | integer | ≥ 0 |
| `true_positive` | integer | ≥ 0 |
| `false_positive` | integer | ≥ 0 |
| `false_negative` | integer | ≥ 0 |

The invariant `true_positive + false_negative == reference_count` must hold. The calculated predicted count is `true_positive + false_positive`.

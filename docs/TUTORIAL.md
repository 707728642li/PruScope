# Five-minute synthetic metric audit

This tutorial verifies the public reviewer path without orchard images or model weights. The fixture is synthetic and tests whether per-image precision, recall, F1, count error, and summary aggregation are computed as documented.

## 1. Run the audit

From the repository root:

```bash
python tools/audit_metrics.py \
  --input examples/synthetic_metric_audit/predictions.csv \
  --output reviewer_check.json
```

PowerShell users may place the command on one line.

## 2. Inspect inputs

`predictions.csv` contains one row per synthetic image:

- `image_id`: non-sensitive synthetic identifier;
- `reference_count`: number of reference objects;
- `true_positive`: predictions matched at the selected IoU rule;
- `false_positive`: unmatched predictions;
- `false_negative`: unmatched references.

The tool rejects negative counts and inconsistent rows where `true_positive + false_negative != reference_count`.

## 3. Inspect outputs

The JSON output reports per-image precision, recall, F1, predicted count, count error, and macro/aggregate summaries. Compare it with:

```bash
python tools/audit_metrics.py \
  --input examples/synthetic_metric_audit/predictions.csv \
  --check examples/synthetic_metric_audit/expected_summary.json
```

A match prints `PASS` and exits with code 0. A mismatch exits nonzero and lists the differing path.

## 4. Run repository checks

```bash
python tools/verify_release.py
python -m unittest discover -s tests -v
```

These checks validate required files, metadata consistency, checksums, the privacy boundary, source syntax, the successful fixture, and one intentional failure fixture.

## Interpretation

Passing this tutorial proves only that the reviewer audit and frozen repository manifest are internally consistent. It does not reproduce neural-network inference because approved model weights and a redistributable image fixture are not yet available.

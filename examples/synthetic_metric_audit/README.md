# Synthetic metric-audit fixture

These rows are fabricated and contain no orchard images, filenames, or observations. The valid fixture exercises clear, dense, empty, and occluded-style count regimes. `invalid_predictions.csv` intentionally violates `TP + FN = reference_count` and must be rejected.

Run:

```bash
python tools/audit_metrics.py --input examples/synthetic_metric_audit/predictions.csv --check examples/synthetic_metric_audit/expected_summary.json
```

The successful output is fixed by `expected_summary.json`; SHA-256 values are recorded in `release/SHA256SUMS`.

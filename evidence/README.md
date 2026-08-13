# Evidence directory

Only aggregate, manuscript-facing evidence that is safe for a private lightweight repository is included. Files containing internal image names, bounding-box coordinates, annotator details, raw predictions, or local absolute paths are excluded.

`metrics/` contains frozen result tables and protocol summaries. The authoritative interpretation is the manuscript plus `docs/MANUSCRIPT_EVIDENCE_MAP.md`; identical metric names from different checkpoints or inference paths must not be substituted.

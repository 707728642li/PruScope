# Reviewer guide

## Start here

1. Read `RELEASE_READINESS.md` to understand what is and is not released.
2. Run the five-minute check in `docs/TUTORIAL.md`.
3. Use `docs/MANUSCRIPT_EVIDENCE_MAP.md` to trace claims to aggregate evidence.
4. Read `MODEL_CARD.md`, `DATA_CARD.md`, and `docs/LIMITATIONS.md` before interpreting scores.
5. Review the current source and PDF in `paper/` and the final figures in `paper/figures/`.

## High-value audit questions

- Are source groups, not derived crops/resolutions, the split and bootstrap units?
- Are validation-selected thresholds and gates clearly separated from protected tests?
- Are three-seed means and negative results reported without best-seed selection?
- Is CitDet described as cross-species generic-fruit evidence rather than a plum-stage test?
- Is DART labelled as post-review known-domain evidence?
- Are no-detection cases retained in strict detector-to-stage performance?
- Do count and stage claims avoid implying physiological maturity, longitudinal tracking, or total yield?
- Are the public-release blockers consistent with the absent weights and data?

## Package limitations

This repository supports manuscript, source, aggregate-evidence, and figure review. It does not permit end-to-end inference reproduction because weights and a redistributable image fixture are not approved. Request confidential artifacts through the journal or corresponding author rather than posting restricted material in an issue.

## Reporting review findings

Reference the commit hash, file, line or table, affected claim, severity, and a reproducible explanation. Do not include local paths, credentials, internal image identifiers, or annotation coordinates in public issues.

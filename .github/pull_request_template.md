## Summary

Describe the focused change and the manuscript claim, user workflow, or release gate it affects.

## Verification

- [ ] `python tools/verify_release.py`
- [ ] `python -m unittest discover -s tests -v`
- [ ] New behavior has a positive and fail-closed test where applicable.
- [ ] Frozen thresholds, splits, test metrics, or evidence labels are unchanged, or the versioned audit impact is explained.
- [ ] No data, annotation, weight, prediction, credential, login-only identifier, or local path was added.

## Reproducibility impact

List changed inputs, outputs, seeds, dependencies, checksums, figures, or manuscript statements.

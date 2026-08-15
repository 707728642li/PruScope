# PruScope release readiness

**Candidate:** `1.0.0-rc.1`  
**Assessment date:** 2026-08-14  
**Current disposition:** suitable for a **private GitHub reviewer repository**; blocked for public release, immutable tagging, DOI minting, package publication, and model-weight distribution.

## Scope reviewed

This gate covers the software-facing repository: core source, architecture configs, aggregate metrics, protocol locks, the homepage architecture overview, documentation, tests, CI, metadata, and checksums. Manuscript files and publication figures are maintained separately. It does not certify a deployable public model because weights and complete redistributable data are absent.

## Completed and verified

- [x] Canonical metadata established in `release/project_metadata.yml`.
- [x] Repository name, candidate version, URL, citation, and changelog aligned.
- [x] Current manuscript title and plum-only terminology aligned.
- [x] Core model/training/inference/evaluation scripts included.
- [x] Reviewed architecture and workflow overview included on the repository homepage.
- [x] Manuscript and publication-figure assets kept outside the software repository.
- [x] Aggregate manuscript evidence and validation locks included.
- [x] Synthetic, redistributable audit example with fixed expected output included.
- [x] Required community, security, support, model-card, and data-card documents included.
- [x] CI uses minimum `contents: read` permission and actions pinned to full commit SHAs.
- [x] Automated check rejects secrets, login-only email, local absolute paths, weights, raw data folders, and oversized files.
- [x] `release/FILE_MANIFEST.csv` and `release/SHA256SUMS` generated from repository contents.

## Public-release blockers (P0)

- [ ] **Code and documentation license:** author/rightsholder must approve an SPDX license. The current `LICENSE` grants no public reuse rights.
- [ ] **Weight redistribution:** base-model license compatibility and institutional rights must be confirmed before weights are deposited.
- [ ] **Internal data/annotation availability:** access conditions, privacy review, and licensing must be approved.
- [ ] **Final authorship metadata:** repository-maintainer identity and associated-manuscript authorship are intentionally separate; final author order and CRediT statement require author approval.
- [ ] **ORCID, funding, and institutional rights:** not confirmed; no values may be inferred.
- [ ] **External-data terms:** the author must reconfirm the versioned upstream licenses and whether any derived files are redistributable.
- [ ] **Clean external model smoke test:** cannot be completed without an approved weight artifact and a redistributable image fixture.

## Author actions before public release

1. Select and approve code/documentation licenses and identify the software rightsholder.
2. Decide whether weights will be public, gated, or available on request; document the compatible base-model license.
3. Approve internal-data and annotation access wording.
4. Confirm authors, order, affiliations, ORCIDs, funding, contributions, conflicts, and acknowledgments.
5. Replace the private license notice, re-run all checks, create an immutable release commit/tag, then create the GitHub Release and archival DOI.
6. Update the manuscript with the exact public tag, commit, version DOI, and final availability statements.

## Explicitly not claimed

- No stable `v1.0.0` tag or GitHub Release exists.
- No Concept DOI or Version DOI exists.
- No PyPI/Conda/container/Hugging Face publication exists.
- Model code availability is not model-weight publication.
- Aggregate evidence availability is not raw-data publication.

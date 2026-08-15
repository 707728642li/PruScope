# Changelog

All notable repository changes are documented here. This project follows Semantic Versioning for reviewer candidates and future releases.

## [Unreleased]

### Changed

- Promoted the reviewed PruScope model workflow to the repository homepage.
- Removed the manuscript, publication figures, and figure-generation scripts from the software repository; these assets are maintained separately.

## [1.0.0-rc.1] - 2026-08-14

### Added

- Private reviewer repository for the PruScope manuscript.
- Frozen manuscript source, review PDF, final main/supplementary figures, and figure scripts.
- Core detector, global-local, ordinal-stage, and DART source code and configurations.
- Aggregate result tables, protocol locks, evidence map, model card, and data card.
- Synthetic metric-audit fixture, integrity tests, CI, file manifest, and SHA-256 checksums.

### Security and release boundary

- Excluded internal images, annotation coordinates, model checkpoints, raw per-image predictions, login identifiers, secrets, and local absolute paths.
- No public reuse license, immutable tag, GitHub Release, package publication, model-hub deposit, or DOI is asserted.

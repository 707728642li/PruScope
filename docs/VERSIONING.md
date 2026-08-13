# Versioning and deprecation policy

PruScope uses Semantic Versioning for repository candidates and releases.

- `1.0.0-rc.N` identifies private/public release candidates before rights and reproduction gates pass.
- `1.0.0` will be the first stable release only after weights/data availability, licensing, clean inference, and metadata are finalized.
- Patch versions correct compatible code or documentation without changing frozen metric definitions.
- Minor versions add backward-compatible functionality or new evidence.
- Major versions may change inputs, outputs, model lineage, metric definitions, or evidence contracts.

Tags and archived releases are immutable. A correction creates a new version; it never moves an old tag. Deprecations will be documented in `CHANGELOG.md` for at least one minor release before removal, unless security or legal risk requires immediate withdrawal.

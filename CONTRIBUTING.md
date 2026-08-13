# Contributing

This repository is a frozen private reviewer candidate. Corrections that improve reproducibility, documentation, tests, or security are welcome through an issue before a pull request is opened.

1. Do not upload orchard images, annotations, checkpoints, raw predictions, credentials, or local paths.
2. Create a focused branch and keep unrelated changes separate.
3. Run `python tools/verify_release.py` and `python -m unittest discover -s tests -v`.
4. Explain which manuscript claim, protocol, or user workflow the change affects.
5. Do not change frozen test results, thresholds, split locks, or evidence labels without a new version and an explicit audit trail.

Contributors must have the right to submit their changes. No contribution may relicense third-party assets or restricted research data.

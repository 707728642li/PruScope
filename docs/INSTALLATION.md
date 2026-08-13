# Installation

## Reviewer audit path

The lightweight integrity and metric audit supports Python 3.10 or later on Windows, Linux, and macOS.

```bash
git clone https://github.com/707728642li/PruScope.git
cd PruScope
python -m venv .venv
```

Activate the environment using your platform's standard command, then run:

```bash
python tools/verify_release.py
python -m unittest discover -s tests -v
```

The audit path is CPU-only and does not download data or weights.

## Full model environment

Full training/inference requires an isolated Conda environment with a CUDA-compatible PyTorch build and the Ultralytics version matching the frozen experiment. Install those components from their official channels for the target operating system and CUDA driver, then install the remaining analysis packages used by the selected script.

The exact historical environment evidence is intentionally not represented as a portable lock because it contained machine-specific paths and GPU packages. A clean, redistributable environment lock will be generated only when weight redistribution and the supported CUDA/PyTorch matrix are approved.

## Offline and proxy environments

The reviewer audit uses only the Python standard library, so it is suitable for offline review. Model weights and datasets are never downloaded implicitly by the repository audit tools.

## Troubleshooting

- **CUDA unavailable** — the reviewer audit still works; model scripts require a matching GPU stack.
- **Dataset path error** — dataset YAMLs are templates containing relative placeholders; obtain approved data and update a local, untracked config.
- **Checkpoint missing** — weights are not distributed in this candidate.
- **Integrity mismatch** — restore the exact commit or regenerate manifests only as part of a reviewed candidate update.

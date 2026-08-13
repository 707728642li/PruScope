# Script reference

PruScope is currently a research pipeline rather than an installed command-line package. Each entry point exposes `--help`; paths and defaults should be recorded in the run ledger.

| Script | Main purpose | Representative required inputs |
|---|---|---|
| `src/train_pruscope_detector.py` | train controlled detector lineages | data YAML, model YAML, seed, image size, epochs |
| `src/predict_pruscope_global_local.py` | direct/GLAF prediction | checkpoint, data/images, confidence, gate settings |
| `src/evaluate_size_stratified.py` | AP/AR by apparent object size | predictions, YOLO references, dataset root |
| `src/aggregate_multiseed_detector.py` | aggregate seed-level detector results | result directories/files |
| `src/bootstrap_detector_comparison_cached.py` | paired image bootstrap | cached predictions from two methods, references, seed, iterations |
| `src/benchmark_detector_runtime.py` | synchronized inference runtime | checkpoint, image list, image size, repeats, warm-up |
| `src/train_stage_ordinal.py` | train DCOH | crop manifest/splits, epochs, seed, geometry flag |
| `src/evaluate_stage_ordinal.py` | reference-region stage evaluation | stage checkpoint, split manifest, intervention mode |
| `src/evaluate_end_to_end_stage.py` | strict detection-to-stage evaluation | detections, DCOH checkpoint, source manifest |
| `src/bootstrap_end_to_end_stage.py` | source-image bootstrap for pipeline outcomes | pipeline results, iterations, seed |
| `src/build_dart_training_manifest.py` | prepare leakage-controlled DART table | locked split/proposals/references |
| `src/train_dart_tail.py` | train DART proposal scorer/refiner | manifest, features, seed, epochs |
| `src/predict_dart_tail.py` | apply frozen DART tail | direct/local predictions, DART checkpoint, lock |
| `src/analyze_dart_density_strata.py` | density-stratified DART comparison | direct/DART predictions and references |
| `tools/audit_metrics.py` | synthetic/public aggregate audit | CSV input; optional output/check file |
| `tools/verify_release.py` | release/privacy/integrity gate | repository root inferred automatically |

## Shared conventions

- Coordinates are floating-point image coordinates unless the script's `--help` says otherwise.
- Seeds are integers and must be recorded before model and loader construction.
- Bootstrap iterations and seeds are explicit.
- Output directories should be new; do not overwrite protected evidence.
- Large training/evaluation artifacts remain outside Git and are referenced by checksums in an approved release ledger.

Run `python <script> --help` for the authoritative parameter list. The repository does not claim a stable public API before `v1.0.0`.

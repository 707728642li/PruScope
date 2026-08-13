# Model configurations

- `pruscope-p2.yaml`: A1, naïve P2 ablation.
- `pruscope-p2h.yaml`: A2, P2 plus capacity-preserving detection head (CCPH).
- `pruscope-m.yaml`: OSSA-enabled lineage used for A5 experiments.
- `yolo26m-p2-fruit.yaml`: earlier P2 fruit architecture retained for lineage audit.

All configurations use one `fruit` class. They depend on Ultralytics modules and the custom registration in `src/pruscope/`. Dataset YAMLs are intentionally absent because public paths and redistribution rights are not finalized.

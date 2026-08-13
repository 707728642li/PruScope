# Limitations and out-of-scope uses

## Scientific limitations

- The collection is cross-sectional and cannot estimate individual fruit growth trajectories.
- Visual cohorts are acquisition-confounded and do not represent calibrated days after flowering.
- No physical fruit diameter, firmness, soluble solids, destructive quality, or harvest-readiness measurements were available.
- External evaluation covers limited archives; additional orchards, seasons, cameras, geographies, and intermediate stages are needed.
- Dense, tiny, foliage-colored, occluded, defocused, shadowed, and boundary-truncated fruit remain the dominant failure regime.
- A strong pooled count correlation can conceal stage-specific undercount and should not replace MAE, bias, and stratified analysis.
- DART improves recall at high latency and does not consistently improve strict AP50–95.

## Engineering limitations

- No public model weights or full environment lock are included.
- Training and model inference require a separately configured GPU environment.
- Some figure scripts depend on restricted images or per-image predictions.
- Research scripts are not yet a stable installed API/CLI package.
- Before stable release, public checkpoint compatibility, small-image smoke inference, and clean-machine reproduction remain mandatory.

## Prohibited interpretations

Do not use uncalibrated output as a sole basis for harvest timing, yield contracts, insurance, pesticide application, safety decisions, or worker performance assessment. Do not describe apparent-size strata as physical measurements or the DCOH index as physiological maturity.

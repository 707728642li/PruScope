# PruScope model card

## Model summary

PruScope is a research pipeline for single-view RGB plum-orchard imagery. A one-class detector localizes visible fruit across small-green, medium-green, and visually mature cohorts. Detected or reference fruit regions can then be assigned ordered stage probabilities by the Developmental Continuum Ordinal Head (DCOH).

The balanced detector lineage is **A2**, which adds stride-4 P2 evidence while preserving the capacity of inherited P3–P5 heads through CCPH. GLAF is a validation-gated tiled-inference option. DART is a slower, offline, high-recall refinement path and does not replace the balanced direct model.

## Intended users and uses

Intended users are plant-phenomics, orchard-vision, and agricultural-robotics researchers who can validate performance on their own cameras and orchards. Intended outputs include visible-fruit locations, counts, apparent-size strata, ordered visual-stage probabilities, and a cross-sectional developmental index.

The model is not intended to determine chronological age, physiological maturity, firmness, soluble solids, total tree yield, food safety, or autonomous harvest safety without additional calibration and human oversight.

## Inputs and outputs

- **Input:** one RGB orchard image; the evaluated detector reference canvas is 1,024 pixels.
- **Detection output:** fruit bounding boxes, confidence scores, and visible count at a frozen operating point.
- **Stage output:** probabilities for `small_green`, `medium_green`, and `visually_mature`, plus an ordinal index from 0 to 2.
- **Coordinate convention:** detector boxes are in the input-image coordinate system after reversing letterbox transforms.

## Architecture and lineage

- **A0:** native detector baseline.
- **A1:** naïve P2 addition.
- **A2:** P2 + CCPH, preserving P3–P5 head capacity.
- **A5:** A2 + Orchard Scale-Selective Attention (OSSA); reported as domain-dependent rather than universally beneficial.
- **GLAF:** validation-frozen selective local inference.
- **DCOH:** ordered stage classifier.
- **DART:** validation-frozen, anchor-preserving local proposal refinement for offline recall-critical analysis.

Architecture definitions are in `src/pruscope/` and `configs/models/`. The underlying detector framework and pretrained starting point are third-party components whose licenses must be reviewed separately before weight redistribution.

## Training and evaluation data

The project used source-group-disjoint plum imagery spanning three cross-sectional visual cohorts. A carefully reviewed 240-image reference contained 5,151 fruit. Public external evidence included an external plum archive and the official CitDet test partition. CitDet was not used for training or parameter selection in the reported zero-tuning experiment.

Images, bounding-box annotations, and per-image predictions are not included in this private-lightweight repository. See `DATA_CARD.md`.

## Frozen evaluation summary

| Evaluation | Result |
|---|---:|
| A2 small AP50–95, internal three-seed mean | 0.433 |
| A2 small AP50–95, external-plum three-seed mean | 0.374 |
| Human audit overall / small AP50–95 | 0.498 / 0.376 |
| Human audit small-green COCO-small AP50–95 / AR50 | 0.205 / 0.340 |
| CitDet zero-tuning COCO-small AP50 / AP50–95 / AR50 | 0.660 / 0.256 / 0.793 |
| DCOH reference-region macro F1 | 0.979 |
| Strict detector-to-stage macro F1 | 0.946 |
| Joint correctly staged fruit recall | 0.459 |

Confidence intervals, seeds, sample sizes, negative results, and protocol roles are in `evidence/metrics/`, the manuscript, and `docs/MANUSCRIPT_EVIDENCE_MAP.md`.

## Efficiency

Direct A2 is the balanced mode. In a separate 30-image CitDet efficiency probe, direct A2 required 20.1 ms/image and full DART 864.0 ms/image on the reported local GPU system. These figures are hardware-, software-, input-, warm-up-, and batch-dependent and should not be treated as universal latency guarantees.

## Limitations and failure modes

- The main remaining error is missed small-green fruit under occlusion, defocus, shadow, and dense branch clutter.
- Apparent-size strata are image-domain measurements, not physical fruit diameter.
- Stage cohorts are cross-sectional and acquisition-confounded; the index is not longitudinal growth.
- External transfer is demonstrated on limited archives, not all plum cultivars, orchards, seasons, cameras, or geographies.
- Aggressive tiling improves recall more reliably than strict localization quality and increases duplicate/ranking risk.
- DART was evaluated on already-known domains and is post-review descriptive evidence.
- Background interventions indicate residual acquisition-context cues in stage classification.

## Risk, ethics, and misuse

The system may systematically undercount dense or occluded canopies. Using uncalibrated counts for commercial harvest, insurance, labor, or resource allocation can create material errors. Users should report operating points, uncertainty, excluded images, and human corrections. Orchard imagery should be reviewed for people, location metadata, property information, and other privacy concerns before sharing.

## Reproducibility and weights

The repository includes source, configurations, aggregate evidence, protocol locks, figure scripts, and checksums. It does not include weights. Weight publication is blocked pending licensing and rightsholder approval. Therefore this candidate supports code/evidence review but not a complete third-party model-inference reproduction.

## Citation and license

See `CITATION.cff`. No public license is granted by this candidate; see `LICENSE` and `RELEASE_READINESS.md`.

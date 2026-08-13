# Post-audit microfruit optimization protocol

Protocol freeze date: 2026-08-11

## Scientific status

The aggregate results from all 240 human-audited images have already been
inspected. This protocol is therefore a transparent post hoc optimization and
cannot create a new independent test set retrospectively. To reduce further
information reuse, T0001-T0160 (annotator A) are assigned to development and
T0161-T0240 (annotator B) are reserved for one final confirmation run. The
confirmation subset has not been evaluated separately before protocol freeze.
Annotator identity is fully confounded with this split and will be disclosed.

## Reviewer-driven hypothesis

The frozen 1536-pixel GLAF tile is downsampled to the detector's 1024-pixel
canvas. This preserves context but can erase microfruit detail. A density-only
gate can also skip images containing few but very small fruit. The intervention
is a selective high-magnification second pass, not an additional trained model.

## Fixed detector and non-tunable components

- Detector: `runs/pruscope_m_a5_stage_strict_gpu0/weights/best.pt`.
- Global canvas: 1024 pixels.
- Prediction floor: 0.001; NMS IoU: 0.70; maximum detections: 500.
- Fusion: existing size-aware weighted box fusion; fusion IoU: 0.55.
- No model weights, labels, or DCOH parameters will be changed.

## Development candidates

The following finite candidate set is fixed before subset-specific evaluation:

1. F0: historical density-60 GLAF, 1536 tile, 25% overlap.
2. F1: always-local GLAF, 1536 tile, 25% overlap.
3. F2: always-local high-magnification GLAF, 1024 tile, 25% overlap.
4. F3: always-local high-magnification GLAF, 768 tile, 25% overlap.
5. F4: always-local high-magnification GLAF, 768 tile, 35% overlap.
6. F5: scale-gated 768-tile GLAF: at least one global candidate at confidence
   0.05 and median 1024-reference box area no greater than 4096 pixels.

Local predictions larger than 4096 reference pixels are excluded for F2-F5 to
focus the second pass on small objects. This cutoff is fixed in advance.

## Selection rule

Evaluate all candidates on the 160-image development subset. A candidate is
eligible only if development overall AP50-95 is no more than 0.01 below F0 and
precision at confidence 0.25 and IoU 0.50 is at least 0.85. Among eligible
candidates, select the highest small-green COCO-small AP50-95. Break ties within
0.002 by lower mean GPU latency, then by fewer local tiles.

After the selection record and its SHA-256 lock are written, run the selected
candidate exactly once on the 80-image confirmation subset. Report all results,
including regressions. Do not retune after confirmation.

## Claim boundary

Any improvement is reported as a post-audit exploratory inference enhancement.
It may motivate a future pre-registered study but cannot replace the manuscript's
frozen 240-image primary analysis or support a fresh independent-test claim.

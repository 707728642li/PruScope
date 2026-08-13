# PruScope-DART microfruit refinement protocol

Protocol frozen: 2026-08-12, before any DART proposal generation, tail
training, validation selection, or DART test inference.

## Scientific status and claim boundary

This is a post-review, protocol-locked structural optimization. The internal
high-resolution, external PLOS plum, CitDet public test, and Human-240 archives
have already been inspected in earlier experiments. They cannot become new
independent confirmation sets retrospectively. DART is selected only on the
existing v4 validation split; subsequent scores on the three protected domains
are reported in full as descriptive, paired post-review evidence. Human-240 is
not evaluated because 159 protected-safe images now contribute to training.

## Evidence motivating the intervention

1. The naive P2 head (A1) reduced accuracy. The capacity-preserving P2+CCPH
   head (A2) repaired all 12 seed-by-domain-by-size comparisons against A1, but
   its three-seed mean was approximately tied with the P3--P5 A0 detector.
   Another ordinary shallow prediction head is therefore not justified.
2. GLAF improved external small-object AP50-95, while MARS raised confirmation
   small-green COCO-small AR50 from 0.355 to 0.759. MARS nevertheless reduced
   overall AP50-95 from 0.591 to 0.577. The unresolved problem is candidate
   precision/localization after recall expansion.
3. Recent tiny-object work supports complementary mechanisms: preserved
   high-resolution features, slicing, background-noise suppression, and
   uncertainty-guided localization refinement. DART tests the last two missing
   elements without replacing the already audited A2 detector.

## Frozen system definition

**DART** means **Density-Aware microfruit Refinement Tail**. It is a trained
post-detector module attached to the selected A2 P2+CCPH localizer.

- Base checkpoint:
  `runs/pruscope_a2_human159x3_s20260805_e30/weights/best.pt`.
- Proposal recall stream: one 1024-pixel global pass plus scale-gated 768-pixel
  overlapping local crops (25% overlap). The local stream is invoked when at
  least one global box at confidence 0.05 is present and its median normalized
  1024-reference area is no greater than 4096 pixels.
- Raw prediction floor 0.001, detector NMS IoU 0.70, at most 500 detections per
  view, and cross-view clustering IoU 0.55.
- DART inputs: an RGB crop around each clustered proposal plus frozen metadata
  describing base confidence, scale, aspect ratio, image position, tile-edge
  reliability, global/local observations, cross-view agreement, and coordinate
  dispersion.
- DART outputs: fruit objectness, four box-refinement offsets, and localization
  uncertainty. Refined proposals undergo one density-preserving duplicate
  suppression pass. Large proposals bypass refinement.

## Training data and leakage control

- Only the 159 manually corrected images proven eligible by
  `reports/data_audit/pruscope_human_240_training_eligibility_v1` are used to
  fit the DART tail. They comprise 120 small-green and 39 mature images with
  detailed labels including blurred and virtual-image fruit.
- The 81 Human-240 medium-green images overlapping the internal test remain
  excluded.
- The v4 validation split is used only for finite model/threshold selection,
  never for gradient updates.
- Protected internal, PLOS external, and CitDet images and labels are not read
  during proposal-tail training or candidate selection.

## Fixed finite development grid

One tail architecture is trained with a deterministic 80/20 group-disjoint
fit/monitor split and then refit on all 159 images for the selected epoch count.
The validation grid is deliberately small:

- DART score fusion weight: 0.25, 0.50, 0.75, or 1.00;
- uncertainty penalty: 0 or 0.25;
- final duplicate IoU: 0.55, 0.65, or 0.75;
- tail bypass area: 1024 or 4096 pixels on the 1024-reference canvas.

All grid points reuse the same cached proposals and tail outputs. No protected
test result may change the grid, weights, epoch, or selected operating point.

## Selection rule and success criteria

Primary selection objective: maximize v4-validation COCO-small AP50-95 subject
to overall AP50-95 being no worse than direct A2 by more than 0.005. Ties within
0.002 small AP50-95 are resolved by higher small AR50, then lower latency.

The structural optimization is considered successful for the manuscript only
if all of the following hold after the frozen selection:

1. v4-validation small AP50-95 exceeds direct A2;
2. at least two of the three protected domains improve COCO-small AP50-95 over
   the identical direct A2 checkpoint;
3. mean protected-domain overall AP50-95 does not decrease by more than 0.005;
4. the result exceeds at least one conventional architecture baseline (A0) on
   the primary small-object endpoint;
5. latency, proposal count, local-stream invocation, and memory are reported;
6. paired image-cluster bootstrap intervals and ablations are retained even if
   they cross zero or expose a regression.

If these conditions fail, DART remains a documented negative experiment and
does not replace the balanced manuscript model.

## Required ablations

- A0 conventional P3--P5 detector;
- A2 P2+CCPH direct detector;
- A2 plus recall stream but without trained DART scoring/refinement (MARS-like);
- DART scoring only;
- DART box refinement only;
- full DART;
- metadata-only tail versus RGB-plus-metadata tail.

## Reporting

Report all-size and COCO-small AP50, AP50-95, AR50, operating-point precision,
recall, F1, count error, latency, peak memory, and density-stratified errors.
Use the terms post-review descriptive evidence and cross-domain robustness;
do not describe any already inspected archive as a fresh independent test.

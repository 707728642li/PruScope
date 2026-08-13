# DART v2 anchor-preserving development addendum

Frozen: 2026-08-12 after the complete v1 validation grid failed its overall
AP50-95 guardrail, and before any DART inference on internal, PLOS, or CitDet
protected tests.

## Retained negative result

All 48 v1 configurations used the trained tail to replace/refine clustered
microfruit proposals. None satisfied the preregistered validation condition
`overall AP50-95 >= direct A2 - 0.005`. For example, D00 raised small AP50-95
from 0.2774 to 0.2787 and AR50 from 0.8526 to 0.9313, but reduced overall
AP50-95 from 0.5878 to 0.5466. The v1 files and diagnosis are retained.

## Diagnosis and one permitted development revision

Recall increased as intended, but replacing global detections changed already
correct medium/large localization and ranking. DART v2 therefore uses an
**anchor-preserving residual policy**:

1. every exact direct A2 global prediction is retained unchanged;
2. DART may contribute only clusters with zero global observations (new
   local-view discoveries) and reference area no greater than the bypass area;
3. additions overlapping a retained global anchor above the duplicate IoU are
   removed; additions are mutually deduplicated with density-preserving NMS;
4. the trained objectness/uncertainty tail ranks additions, while direct-anchor
   scores and coordinates are never altered;
5. v1 evidence showed learned box replacement caused a localization trade-off,
   so the primary v2 grid disables box offsets. Box refinement remains a
   reported negative ablation rather than being silently removed.

## Frozen v2 validation grid

- score fusion weight: 0.25, 0.50, 0.75, or 1.00;
- uncertainty penalty: 0 or 0.25;
- addition duplicate IoU: 0.55, 0.65, or 0.75;
- maximum addition reference area: 1024 or 4096 pixels;
- box offset: disabled for all primary candidates.

Selection and success rules are unchanged from the parent protocol. If no v2
candidate passes the overall guardrail, DART is rejected without protected-test
inference. No further structural iteration is permitted from this validation
archive in the present study.

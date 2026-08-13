# Frozen public-external CitDet validation protocol

Protocol freeze date: 2026-08-11

## Scientific purpose and status

This analysis tests transportability to a public, independently acquired fruit
domain after all internal and PLOS-plum results have been inspected. The public
CitDet dataset was selected before downloading images or evaluating PruScope
because it provides human bounding boxes, an official held-out test archive,
high-resolution orchard scenes, multiple acquisition dates and tree sides, more
than 60 citrus varieties, and fruit spanning different apparent colors and
sizes. It has not contributed images, labels, pseudo-labels, model weights, or
threshold choices to PruScope.

The analysis is confirmatory with respect to the frozen PruScope operating
points but external to the plum target species. It tests generic fruit
localization and apparent-size transportability; it does not test plum
developmental-stage classification.

## Data lock

- Dataset: CitDet, DOI `10.32855/dataset.2024.05.005`; mirrored data DOI
  `10.18738/T8/QFVHQ5`.
- License: CC BY-NC-SA 4.0; use is restricted to this non-commercial academic
  validation.
- Evaluation split: every image and human bounding box in the official
  `CitDet-test.zip` archive; the training archive is not used.
- Category handling: all annotated citrus fruit categories are mapped to the
  single generic class `fruit`; both tree and ground fruit remain included.
- No image, annotation, or subset will be removed after predictions are seen.
- Exact archive MD5, file inventory, COCO-to-YOLO conversion manifest, and an
  image-overlap audit against every PruScope training/validation/test source will
  be stored before evaluation.

## Frozen model and operating points

All modes use
`runs/pruscope_m_a5_stage_strict_gpu0/weights/best.pt`, a 1,024-pixel global
canvas, prediction floor 0.001, NMS IoU 0.70, fusion IoU 0.55, and maximum 500
detections per image. No weights, boxes, thresholds, or fusion parameters will
be changed.

1. `D0-direct`: global 1,024-pixel inference only.
2. `G0-primary`: the manuscript's validation-frozen GLAF policy: 1,536-pixel
   tiles with 25% overlap; invoke the local stream when at least five global
   candidates at confidence 0.05 have median 1,024-reference area no greater
   than 2,048 pixels.
3. `MARS`: the already locked recall-priority policy: 768-pixel tiles with 25%
   overlap; invoke the local stream when at least one global candidate at
   confidence 0.05 has median 1,024-reference area no greater than 4,096 pixels;
   reject local boxes larger than 4,096 reference pixels.

## Endpoints fixed before evaluation

Primary endpoints for each operating point are overall AP50-95, AP50, and AR50,
plus COCO-small AP50-95, AP50, and AR50 after projecting boxes to a 1,024 x
1,024 reference canvas. Secondary endpoints are medium/large size strata,
precision/recall/F1 at confidence 0.25 and IoU 0.50, visible-fruit count MAE and
bias at the same threshold, local-stream invocation rate, and mean end-to-end
latency.

Paired MARS-minus-G0 and G0-minus-D0 differences will use 2,000 source-image
cluster bootstrap replicates. Confidence intervals are two-sided percentile 95%
intervals. The complete metric table and regressions will be reported.

## Interpretation rule

No minimum performance threshold is required for reporting. Evidence of
transportability requires an acceptable absolute external result, not merely a
positive paired difference between inference modes. MARS will be described as
recall-priority only if recall increases without implying improved AP50-95 when
its confidence interval includes zero or the point estimate decreases. CitDet
will not be used for tuning or training after this one evaluation.

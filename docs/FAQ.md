# FAQ and troubleshooting

## Why are weights absent?

Their redistribution depends on institutional rights and compatibility with the pretrained base-model license. Code availability must not be misreported as model publication.

## Why are orchard images and annotations absent?

Internal redistribution and privacy conditions are not finalized. External datasets remain at their source records under upstream terms.

## Can I run the tests without a GPU?

Yes. The reviewer audit and synthetic example are CPU-only. Neural inference/training is outside this lightweight test path.

## Which model is the final model?

Direct A2 is the balanced detector. GLAF is selectively invoked under a frozen gate. DART is an optional offline high-recall tail. A5/OSSA is retained as a controlled, domain-dependent experiment rather than a universal improvement.

## Does the stage index measure maturity or time?

No. It orders three cross-sectional visual cohorts. Physiological maturity and longitudinal growth require new measurements.

## Why can AP50 increase while AP50–95 does not?

Small objects may be found at a permissive overlap threshold while box coordinates and ranking remain inadequate at stricter IoU thresholds. Tiling can amplify this difference.

## An integrity check failed. What should I do?

Confirm you are on the recorded commit and did not edit tracked artifacts. Candidate updates require regenerating the manifest and checksums, rerunning tests, and committing the change as a reviewed version update.

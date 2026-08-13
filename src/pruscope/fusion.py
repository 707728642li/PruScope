"""Global-local inference utilities for PruScope."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import box_iou, nms


@dataclass(frozen=True)
class ImageTile:
    """A half-open crop window in full-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    stride = max(1, round(window * (1.0 - overlap)))
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_overlapping_tiles(
    image_width: int,
    image_height: int,
    tile_size: int = 1536,
    overlap: float = 0.25,
) -> list[ImageTile]:
    """Cover an image exactly with deterministic overlapping square tiles."""
    if image_width < 1 or image_height < 1 or tile_size < 1:
        raise ValueError("image dimensions and tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    tile_width = min(tile_size, image_width)
    tile_height = min(tile_size, image_height)
    return [
        ImageTile(x, y, x + tile_width, y + tile_height)
        for y in _axis_starts(image_height, tile_height, overlap)
        for x in _axis_starts(image_width, tile_width, overlap)
    ]


def tile_edge_reliability(
    local_boxes: torch.Tensor,
    tile: ImageTile,
    image_width: int,
    image_height: int,
    margin_fraction: float = 0.04,
) -> torch.Tensor:
    """Downweight boxes truncated near artificial internal tile boundaries."""
    if local_boxes.ndim != 2 or local_boxes.shape[1] != 4:
        raise ValueError("local_boxes must have shape [N,4]")
    if len(local_boxes) == 0:
        return local_boxes.new_empty((0,))
    margin = max(1.0, min(tile.width, tile.height) * margin_fraction)
    distances = torch.stack(
        (
            local_boxes[:, 0],
            local_boxes[:, 1],
            tile.width - local_boxes[:, 2],
            tile.height - local_boxes[:, 3],
        ),
        dim=1,
    )
    internal = local_boxes.new_tensor(
        [
            tile.x1 > 0,
            tile.y1 > 0,
            tile.x2 < image_width,
            tile.y2 < image_height,
        ]
    ).bool()
    if not internal.any():
        return local_boxes.new_ones((len(local_boxes),))
    normalized = (distances[:, internal] / margin).clamp(0.0, 1.0)
    # A smooth floor prevents a plausible boundary fruit from being discarded;
    # overlapping neighbours can still recover and reinforce it.
    return 0.35 + 0.65 * normalized.amin(dim=1)


def adaptive_stream_weight(
    boxes: torch.Tensor,
    image_width: int,
    image_height: int,
    stream: str,
    reference_size: int = 1024,
    transition_area: float = 32.0**2,
) -> torch.Tensor:
    """Prefer local tiles for microfruit and the full image for larger fruit."""
    if stream not in {"global", "local"}:
        raise ValueError("stream must be 'global' or 'local'")
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
    reference_area = (
        wh[:, 0]
        * wh[:, 1]
        / float(image_width * image_height)
        * float(reference_size**2)
    )
    # Smoothly changes preference around the COCO-equivalent small-object
    # boundary instead of applying a brittle hard size cutoff.
    microfruit_probability = torch.sigmoid(
        (transition_area - reference_area) / (0.35 * transition_area)
    )
    if stream == "local":
        return 0.90 + 0.30 * microfruit_probability
    return 1.10 - 0.20 * microfruit_probability


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = (box[2] - box[0]).clamp_min(0.0) * (box[3] - box[1]).clamp_min(0.0)
    boxes_wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
    boxes_area = boxes_wh[:, 0] * boxes_wh[:, 1]
    union = box_area + boxes_area - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def size_aware_weighted_box_fusion(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    reliability: torch.Tensor,
    iou_threshold: float = 0.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse global/local detections using confidence and reliability weights.

    Returns fused boxes, confidence scores, and the number of observations in
    each cluster. It is class-agnostic because PruScope localizes one generic
    fruit class; the DCOH stage probabilities are attached after localization.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N,4]")
    if scores.shape != (len(boxes),) or reliability.shape != (len(boxes),):
        raise ValueError("scores and reliability must have shape [N]")
    if len(boxes) == 0:
        return boxes, scores, scores.new_empty((0,), dtype=torch.long)
    effective = (scores * reliability).clamp(0.0, 1.0)
    # Compiled NMS provides deterministic cluster seeds in score order. Every
    # suppressed observation is then assigned to its highest-IoU surviving
    # seed, after which index_add performs all weighted aggregation in tensors.
    # This preserves the original fusion definition while avoiding a Python
    # O(N^2) loop on extremely dense low-confidence orchard predictions.
    seed_indices = nms(boxes.float(), effective.float(), iou_threshold)
    seed_boxes = boxes[seed_indices]
    overlaps = box_iou(boxes.float(), seed_boxes.float())
    best_overlap, assignments = overlaps.max(dim=1)
    if torch.any(best_overlap < iou_threshold):
        raise RuntimeError("NMS produced an observation without a matching seed")
    cluster_count = len(seed_indices)
    coordinate_weights = effective.clamp_min(1e-6)
    coordinate_sums = boxes.new_zeros((cluster_count, 4))
    coordinate_sums.index_add_(0, assignments, boxes * coordinate_weights[:, None])
    weight_sums = scores.new_zeros((cluster_count,))
    weight_sums.index_add_(0, assignments, coordinate_weights)
    fused_boxes = coordinate_sums / weight_sums[:, None].clamp_min(1e-6)

    confidence_sums = scores.new_zeros((cluster_count,))
    confidence_sums.index_add_(0, assignments, scores * reliability)
    reliability_sums = scores.new_zeros((cluster_count,))
    reliability_sums.index_add_(0, assignments, reliability)
    fused_scores = confidence_sums / reliability_sums.clamp_min(1e-6)
    observations = torch.bincount(assignments, minlength=cluster_count)
    return fused_boxes, fused_scores.clamp(0.0, 1.0), observations

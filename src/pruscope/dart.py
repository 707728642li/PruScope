"""Density-Aware microfruit Refinement Tail (DART) utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.ops import box_iou, nms


DART_METADATA_NAMES = (
    "base_confidence_logit",
    "max_confidence_logit",
    "global_confidence_logit",
    "local_confidence_logit",
    "log_observations",
    "local_observation_fraction",
    "mean_reliability",
    "score_std",
    "one_minus_mean_cluster_iou",
    "box_dispersion",
    "log_reference_area",
    "log_aspect_ratio",
    "center_x",
    "center_y",
    "boundary_proximity",
    "log_image_density",
)


def _scatter_sum(values: torch.Tensor, assignments: torch.Tensor, count: int) -> torch.Tensor:
    output = values.new_zeros((count,) + values.shape[1:])
    output.index_add_(0, assignments, values)
    return output


def _scatter_max(values: torch.Tensor, assignments: torch.Tensor, count: int) -> torch.Tensor:
    output = values.new_full((count,), float("-inf"))
    output.scatter_reduce_(0, assignments, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


def cluster_multiview_proposals(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    reliability: torch.Tensor,
    streams: torch.Tensor,
    iou_threshold: float = 0.55,
) -> dict[str, torch.Tensor]:
    """Fuse multi-view boxes and preserve DART uncertainty/agreement statistics.

    ``streams`` is zero for the full-image view and one for local tiles. The
    function is single-class by design because PruScope first localizes generic
    fruit and assigns developmental stage afterward.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N,4]")
    if any(value.shape != (len(boxes),) for value in (scores, reliability, streams)):
        raise ValueError("scores, reliability, and streams must have shape [N]")
    if len(boxes) == 0:
        empty = boxes.new_empty((0,))
        return {
            "boxes": boxes,
            "scores": empty,
            "observations": empty.long(),
            "global_observations": empty.long(),
            "local_observations": empty.long(),
            "max_score": empty,
            "global_max_score": empty,
            "local_max_score": empty,
            "mean_reliability": empty,
            "score_std": empty,
            "mean_cluster_iou": empty,
            "box_dispersion": empty,
        }

    boxes = boxes.float()
    scores = scores.float().clamp(0.0, 1.0)
    reliability = reliability.float().clamp_min(0.0)
    streams = streams.long()
    effective = (scores * reliability).clamp(0.0, 1.0)
    seed_indices = nms(boxes, effective, iou_threshold)
    seed_boxes = boxes[seed_indices]
    overlaps = box_iou(boxes, seed_boxes)
    best_overlap, assignments = overlaps.max(dim=1)
    if torch.any(best_overlap + 1e-6 < iou_threshold):
        raise RuntimeError("NMS produced an observation without a matching cluster seed")
    cluster_count = len(seed_indices)

    coordinate_weights = effective.clamp_min(1e-6)
    weight_sums = _scatter_sum(coordinate_weights, assignments, cluster_count)
    fused_boxes = _scatter_sum(
        boxes * coordinate_weights[:, None], assignments, cluster_count
    ) / weight_sums[:, None].clamp_min(1e-6)
    reliability_sums = _scatter_sum(reliability, assignments, cluster_count)
    fused_scores = _scatter_sum(
        scores * reliability, assignments, cluster_count
    ) / reliability_sums.clamp_min(1e-6)

    observations = torch.bincount(assignments, minlength=cluster_count)
    global_mask = streams == 0
    local_mask = ~global_mask
    global_observations = torch.bincount(
        assignments[global_mask], minlength=cluster_count
    )
    local_observations = observations - global_observations
    max_score = _scatter_max(scores, assignments, cluster_count)
    global_max_score = _scatter_max(
        scores[global_mask], assignments[global_mask], cluster_count
    ) if global_mask.any() else scores.new_zeros(cluster_count)
    local_max_score = _scatter_max(
        scores[local_mask], assignments[local_mask], cluster_count
    ) if local_mask.any() else scores.new_zeros(cluster_count)
    mean_reliability = reliability_sums / observations.clamp_min(1)
    score_mean = _scatter_sum(scores, assignments, cluster_count) / observations.clamp_min(1)
    score_variance = _scatter_sum(
        (scores - score_mean[assignments]).square(), assignments, cluster_count
    ) / observations.clamp_min(1)

    fused_overlaps = box_iou(boxes, fused_boxes)
    assigned_iou = fused_overlaps[torch.arange(len(boxes)), assignments]
    mean_cluster_iou = _scatter_sum(
        assigned_iou, assignments, cluster_count
    ) / observations.clamp_min(1)

    obs_center = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    obs_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1.0)
    fused_center = (fused_boxes[:, :2] + fused_boxes[:, 2:]) / 2.0
    fused_size = (fused_boxes[:, 2:] - fused_boxes[:, :2]).clamp_min(1.0)
    normalized_deviation = torch.cat(
        (
            (obs_center - fused_center[assignments]) / fused_size[assignments],
            torch.log(obs_size / fused_size[assignments]),
        ),
        dim=1,
    )
    box_dispersion = torch.sqrt(
        _scatter_sum(
            normalized_deviation.square().mean(dim=1), assignments, cluster_count
        )
        / observations.clamp_min(1)
    )
    return {
        "boxes": fused_boxes,
        "scores": fused_scores.clamp(0.0, 1.0),
        "observations": observations,
        "global_observations": global_observations,
        "local_observations": local_observations,
        "max_score": max_score,
        "global_max_score": global_max_score,
        "local_max_score": local_max_score,
        "mean_reliability": mean_reliability,
        "score_std": torch.sqrt(score_variance.clamp_min(0.0)),
        "mean_cluster_iou": mean_cluster_iou,
        "box_dispersion": box_dispersion,
    }


def safe_logit(value: float, epsilon: float = 1e-5) -> float:
    value = min(max(float(value), epsilon), 1.0 - epsilon)
    return math.log(value / (1.0 - value))


def proposal_metadata_vector(
    proposal: dict,
    width: int,
    height: int,
    image_density: int,
    reference_size: int = 1024,
) -> list[float]:
    """Create the ordered, raw metadata vector consumed by DART."""
    x1, y1, x2, y2 = map(float, proposal["xyxy"])
    box_width = max(x2 - x1, 1e-3)
    box_height = max(y2 - y1, 1e-3)
    reference_area = box_width * box_height / max(width * height, 1) * reference_size**2
    center_x = (x1 + x2) / (2.0 * max(width, 1))
    center_y = (y1 + y2) / (2.0 * max(height, 1))
    boundary_proximity = min(x1 / width, y1 / height, (width - x2) / width, (height - y2) / height)
    observations = max(int(proposal["observations"]), 1)
    return [
        safe_logit(proposal["confidence"]),
        safe_logit(proposal["max_confidence"]),
        safe_logit(proposal.get("global_max_confidence", 0.0)),
        safe_logit(proposal.get("local_max_confidence", 0.0)),
        math.log1p(observations),
        float(proposal.get("local_observations", 0)) / observations,
        float(proposal["mean_reliability"]),
        float(proposal["score_std"]),
        1.0 - float(proposal["mean_cluster_iou"]),
        float(proposal["box_dispersion"]),
        math.log1p(max(reference_area, 0.0)),
        math.log(box_width / box_height),
        center_x,
        center_y,
        boundary_proximity,
        math.log1p(max(image_density, 0)),
    ]


def encode_box_delta(proposal: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Encode target box relative to proposal as center/scale offsets."""
    proposal_size = (proposal[2:] - proposal[:2]).clamp_min(1e-3)
    proposal_center = (proposal[:2] + proposal[2:]) / 2.0
    target_size = (target[2:] - target[:2]).clamp_min(1e-3)
    target_center = (target[:2] + target[2:]) / 2.0
    return torch.cat(
        ((target_center - proposal_center) / proposal_size, torch.log(target_size / proposal_size))
    )


def decode_box_delta(
    proposal: torch.Tensor,
    delta: torch.Tensor,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor:
    """Apply center/scale offsets, optionally clipping to an image canvas."""
    proposal_size = (proposal[..., 2:] - proposal[..., :2]).clamp_min(1e-3)
    proposal_center = (proposal[..., :2] + proposal[..., 2:]) / 2.0
    delta = torch.cat((delta[..., :2].clamp(-1.0, 1.0), delta[..., 2:].clamp(-1.5, 1.5)), dim=-1)
    target_center = proposal_center + delta[..., :2] * proposal_size
    target_size = proposal_size * torch.exp(delta[..., 2:])
    output = torch.cat((target_center - target_size / 2.0, target_center + target_size / 2.0), dim=-1)
    if width is not None and height is not None:
        output[..., (0, 2)] = output[..., (0, 2)].clamp(0.0, float(width))
        output[..., (1, 3)] = output[..., (1, 3)].clamp(0.0, float(height))
    return output


def density_preserving_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """DIoU-NMS: suppress duplicate views while retaining adjacent fruit."""
    if len(boxes) == 0:
        return boxes.new_empty((0,), dtype=torch.long)
    order = torch.argsort(scores, descending=True, stable=True)
    keep: list[torch.Tensor] = []
    while len(order):
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        overlap = box_iou(boxes[current][None], boxes[remaining])[0]
        current_center = (boxes[current, :2] + boxes[current, 2:]) / 2.0
        remaining_center = (boxes[remaining, :2] + boxes[remaining, 2:]) / 2.0
        center_distance = (remaining_center - current_center).square().sum(dim=1)
        enclosing_tl = torch.minimum(boxes[current, :2], boxes[remaining, :2])
        enclosing_br = torch.maximum(boxes[current, 2:], boxes[remaining, 2:])
        enclosing_diagonal = (enclosing_br - enclosing_tl).square().sum(dim=1).clamp_min(1e-6)
        diou = overlap - center_distance / enclosing_diagonal
        order = remaining[diou <= threshold]
    return torch.stack(keep)


class PruScopeDARTTail(nn.Module):
    """RGB-plus-metadata proposal objectness and localization refiner."""

    def __init__(self, metadata_channels: int = len(DART_METADATA_NAMES)) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        visual_channels = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.visual_encoder = backbone
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_channels, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.15)
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, 96), nn.SiLU(),
        )
        self.objectness = nn.Linear(96, 1)
        self.box_delta = nn.Linear(96, 4)
        self.log_variance = nn.Linear(96, 4)

    def forward(
        self,
        crops: torch.Tensor,
        metadata: torch.Tensor,
        zero_visual: bool = False,
        zero_metadata: bool = False,
    ) -> dict[str, torch.Tensor]:
        if zero_visual:
            visual = crops.new_zeros((len(crops), 128))
        else:
            visual = self.visual_projection(self.visual_encoder(crops))
        if zero_metadata:
            encoded_metadata = metadata.new_zeros((len(metadata), 64))
        else:
            encoded_metadata = self.metadata_encoder(metadata)
        fused = self.fusion(torch.cat((visual, encoded_metadata), dim=1))
        return {
            "objectness_logit": self.objectness(fused).squeeze(1),
            "box_delta": self.box_delta(fused),
            "log_variance": self.log_variance(fused).clamp(-5.0, 3.0),
        }

    def load_visual_encoder(self, state: dict[str, torch.Tensor], prefix: str = "visual_encoder.") -> int:
        visual_state = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        self.visual_encoder.load_state_dict(visual_state, strict=True)
        return len(visual_state)


@dataclass(frozen=True)
class DARTSelection:
    score_weight: float
    uncertainty_penalty: float
    nms_iou: float
    bypass_area: float

"""Neural-network modules developed for PruScope."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class _DilatedDepthwiseBranch(nn.Module):
    """Efficient local-context branch with a prescribed receptive field."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OrchardScaleSelectiveAttention(nn.Module):
    """Select local context scales while suppressing orchard-background clutter.

    Three depthwise branches provide 3x3, 5x5, and 7x7 effective receptive
    fields. A per-channel selector weights the branches, after which a spatial
    gate emphasizes compact fruit-like regions. The residual layer scale starts
    at zero so a detector can be initialized from pretrained YOLO features
    without an abrupt distribution shift.

    The module preserves the number of channels and spatial resolution, which
    makes it suitable for the high-resolution P2 and P3 pyramid levels.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if spatial_kernel not in {3, 7}:
            raise ValueError("spatial_kernel must be 3 or 7")
        hidden = max(channels // reduction, 16)
        self.channels = channels
        self.branches = nn.ModuleList(
            [_DilatedDepthwiseBranch(channels, dilation) for dilation in (1, 2, 3)]
        )
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * len(self.branches), kernel_size=1, bias=True),
        )
        padding = spatial_kernel // 2
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        projection_groups = 8 if channels % 8 == 0 else 1
        self.projection = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                groups=projection_groups,
                bias=False,
            ),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03),
        )
        self.layer_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        selector = self.scale_selector(branch_features.sum(dim=1))
        batch, _, _, _ = selector.shape
        selector = selector.view(batch, len(self.branches), self.channels, 1, 1)
        selector = selector.softmax(dim=1)
        fused = (branch_features * selector).sum(dim=1)
        spatial_statistics = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)], dim=1
        )
        fused = fused * self.spatial_gate(spatial_statistics)
        return x + self.layer_scale * self.projection(fused)


class CrossScaleCapacityPreservingDetect(Detect):
    """P2-P5 head that does not let the narrow P2 level shrink every branch.

    Ultralytics Detect derives the hidden widths of all detection branches from
    the first input feature. After a P2 level is prepended, this makes the P3-P5
    heads substantially narrower than their three-level counterparts. PruScope
    instead derives shared hidden widths from a configurable reference level
    (P3 by default), retaining semantic capacity for medium and mature fruit
    while adding a dedicated high-resolution P2 branch for microfruit.
    """

    def __init__(
        self,
        nc: int = 1,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple[int, ...] = (),
        reference_level: int = 1,
    ) -> None:
        nn.Module.__init__(self)
        if not ch:
            raise ValueError("ch must contain the P2-P5 input channels")
        if not 0 <= reference_level < len(ch):
            raise ValueError("reference_level is outside the feature pyramid")
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        reference_channels = ch[reference_level]
        box_channels = max(16, reference_channels // 4, self.reg_max * 4)
        class_channels = max(reference_channels, min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(input_channels, box_channels, 3),
                Conv(box_channels, box_channels, 3),
                nn.Conv2d(box_channels, 4 * self.reg_max, 1),
            )
            for input_channels in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    DWConv(input_channels, input_channels, 3),
                    Conv(input_channels, class_channels, 1),
                ),
                nn.Sequential(
                    DWConv(class_channels, class_channels, 3),
                    Conv(class_channels, class_channels, 1),
                ),
                nn.Conv2d(class_channels, self.nc, 1),
            )
            for input_channels in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self._end2end = end2end
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)


class DevelopmentalContinuumOrdinalHead(nn.Module):
    """Rank-consistent head for biologically ordered fruit stages.

    For three stages the head estimates two cumulative probabilities,
    ``P(stage > small_green)`` and ``P(stage > medium_green)``. Ordered,
    learnable cut points guarantee that the second probability cannot exceed
    the first. Their sum is the expected stage and provides a continuous
    developmental index in addition to discrete stage probabilities.

    The head accepts either pooled ``[N, C]`` ROI features or spatial
    ``[N, C, H, W]`` crop/ROI features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_stages: int = 3,
        dropout: float = 0.15,
        minimum_cutpoint_gap: float = 0.05,
        geometry_channels: int = 3,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if num_stages < 2:
            raise ValueError("num_stages must be at least two")
        if minimum_cutpoint_gap <= 0:
            raise ValueError("minimum_cutpoint_gap must be positive")
        self.num_stages = num_stages
        self.geometry_channels = geometry_channels
        self.minimum_cutpoint_gap = minimum_cutpoint_gap
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(in_channels + geometry_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.stage_score = nn.Linear(hidden_channels, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-0.5))
        if num_stages > 2:
            initial_gap = 1.0 - minimum_cutpoint_gap
            inverse_softplus = math.log(math.expm1(initial_gap))
            self.unconstrained_gaps = nn.Parameter(
                torch.full((num_stages - 2,), inverse_softplus)
            )
        else:
            self.register_parameter("unconstrained_gaps", None)

    def cutpoints(self) -> torch.Tensor:
        """Return strictly increasing developmental cut points."""
        first = self.first_cutpoint.reshape(1)
        if self.unconstrained_gaps is None:
            return first
        positive_gaps = (
            torch.nn.functional.softplus(self.unconstrained_gaps)
            + self.minimum_cutpoint_gap
        )
        return torch.cat((first, first + positive_gaps.cumsum(dim=0)))

    def forward(
        self,
        features: torch.Tensor,
        geometry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        elif features.ndim != 2:
            raise ValueError("features must have shape [N,C] or [N,C,H,W]")
        if geometry is None:
            geometry = features.new_zeros((len(features), self.geometry_channels))
        if geometry.shape != (len(features), self.geometry_channels):
            raise ValueError(
                f"geometry must have shape [N,{self.geometry_channels}]"
            )
        features = torch.cat((features, geometry.to(features.dtype)), dim=1)
        score = self.stage_score(self.embedding(features))
        cumulative_logits = score - self.cutpoints().reshape(1, -1)
        cumulative_probabilities = cumulative_logits.sigmoid()
        stage_probabilities = torch.cat(
            (
                1.0 - cumulative_probabilities[:, :1],
                cumulative_probabilities[:, :-1] - cumulative_probabilities[:, 1:],
                cumulative_probabilities[:, -1:],
            ),
            dim=1,
        )
        return {
            "cumulative_logits": cumulative_logits,
            "cumulative_probabilities": cumulative_probabilities,
            "stage_probabilities": stage_probabilities,
            "developmental_index": cumulative_probabilities.sum(dim=1),
        }

    def loss(
        self,
        cumulative_logits: torch.Tensor,
        stage_targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Binary cumulative-link loss for integer targets in ``[0, K-1]``."""
        if stage_targets.ndim != 1:
            raise ValueError("stage_targets must be one-dimensional")
        if cumulative_logits.shape[0] != stage_targets.shape[0]:
            raise ValueError("batch dimension differs between logits and targets")
        if torch.any((stage_targets < 0) | (stage_targets >= self.num_stages)):
            raise ValueError("stage target is outside the configured range")
        rank = torch.arange(
            self.num_stages - 1,
            device=stage_targets.device,
        ).reshape(1, -1)
        cumulative_targets = (stage_targets.reshape(-1, 1) > rank).to(
            cumulative_logits.dtype
        )
        return torch.nn.functional.binary_cross_entropy_with_logits(
            cumulative_logits,
            cumulative_targets,
            reduction=reduction,
        )

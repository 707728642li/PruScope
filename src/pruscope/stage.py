"""Lightweight ROI stage branch for the complete PruScope framework."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .modules import DevelopmentalContinuumOrdinalHead


class PruScopeROIStageModel(nn.Module):
    """Encode detected fruit crops and infer an ordered developmental state.

    The visual branch captures colour, surface texture, and fruit morphology.
    Three geometry features (log normalized width, height, and area) couple
    apparent scale to the visual representation without making scale alone the
    stage decision. The rank-consistent DCOH produces both three named stages
    and a continuous developmental index.
    """

    def __init__(
        self,
        pretrained: bool = False,
        hidden_channels: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_channels = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_channels = feature_channels
        self.visual_encoder = backbone
        self.ordinal_head = DevelopmentalContinuumOrdinalHead(
            in_channels=feature_channels,
            hidden_channels=hidden_channels,
            num_stages=3,
            dropout=dropout,
            geometry_channels=3,
        )

    def forward(
        self,
        crops: torch.Tensor,
        geometry: torch.Tensor,
        zero_visual: bool = False,
    ) -> dict[str, torch.Tensor]:
        if zero_visual:
            visual_features = crops.new_zeros((len(crops), self.feature_channels))
        else:
            visual_features = self.visual_encoder(crops)
        return self.ordinal_head(visual_features, geometry)

"""PruScope: whole-development-cycle Prunus fruit phenotyping."""

from .modules import (
    CrossScaleCapacityPreservingDetect,
    DevelopmentalContinuumOrdinalHead,
    OrchardScaleSelectiveAttention,
)
from .fusion import (
    ImageTile,
    adaptive_stream_weight,
    generate_overlapping_tiles,
    size_aware_weighted_box_fusion,
    tile_edge_reliability,
)
from .stage import PruScopeROIStageModel
from .dart import (
    DART_METADATA_NAMES,
    DARTSelection,
    PruScopeDARTTail,
    cluster_multiview_proposals,
    decode_box_delta,
    density_preserving_nms,
    encode_box_delta,
    proposal_metadata_vector,
)
from .register import activate_pruscope_detect_head, register_pruscope_modules

__all__ = [
    "CrossScaleCapacityPreservingDetect",
    "DevelopmentalContinuumOrdinalHead",
    "OrchardScaleSelectiveAttention",
    "activate_pruscope_detect_head",
    "register_pruscope_modules",
    "ImageTile",
    "adaptive_stream_weight",
    "generate_overlapping_tiles",
    "size_aware_weighted_box_fusion",
    "tile_edge_reliability",
    "PruScopeROIStageModel",
    "DART_METADATA_NAMES",
    "DARTSelection",
    "PruScopeDARTTail",
    "cluster_multiview_proposals",
    "decode_box_delta",
    "density_preserving_nms",
    "encode_box_delta",
    "proposal_metadata_vector",
]

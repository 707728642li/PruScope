"""Register PruScope modules with the Ultralytics YAML model parser."""

from __future__ import annotations

from ultralytics.nn import tasks

from .modules import CrossScaleCapacityPreservingDetect, OrchardScaleSelectiveAttention


def register_pruscope_modules() -> None:
    """Expose custom modules without modifying the installed Ultralytics package."""
    tasks.OrchardScaleSelectiveAttention = OrchardScaleSelectiveAttention
    tasks.CrossScaleCapacityPreservingDetect = CrossScaleCapacityPreservingDetect


def activate_pruscope_detect_head() -> None:
    """Use the capacity-preserving subclass for YAML `Detect` layers.

    Ultralytics constructs its parser's recognized-head set locally from the
    global `Detect` symbol. Temporarily selecting the subclass through that
    symbol preserves all native parser behavior, including automatic channel
    injection and stride initialization. PruScope training runs in a dedicated
    process, so the selection is process-local and cannot affect other jobs.
    """
    tasks.Detect = CrossScaleCapacityPreservingDetect


register_pruscope_modules()

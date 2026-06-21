from __future__ import annotations

from plant_3d.application.service import Plant3DService
from plant_3d.infrastructure.adapters import (
    CloudReaderAdapter,
    LightingAdapter,
    MorphometricsAdapter,
    RegistrationAdapter,
    SegmentationAdapter,
    TrackingAdapter,
)
from plant_3d.infrastructure.hunyuan_adapter import HunyuanAdapter
from plant_3d.infrastructure.rembg_adapter import RembgAdapter


def build_default_service() -> Plant3DService:
    return Plant3DService(
        preprocess=RembgAdapter(),
        generation=HunyuanAdapter(),
        segmentation=SegmentationAdapter(),
        registration=RegistrationAdapter(),
        lighting=LightingAdapter(),
        clouds=CloudReaderAdapter(),
        tracking=TrackingAdapter(),
        morph=MorphometricsAdapter(),
    )

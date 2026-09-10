from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.models import ModelProvenance
from satquery.schemas.segmentation import SegmentationResult
from satquery.tools.base import BaseTool


class SegmentationTool(BaseTool):
    """Tool for pixel-level land cover segmentation and exact area percentage calculations."""

    def __init__(self):
        super().__init__(
            name="SegmentationTool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["segmentation", "segmentation_area"],
            provenance=ModelProvenance.HEURISTIC,
            description="Agentic tool calculating verified pixel mask coverage and area metrics",
        )

    def execute(
        self,
        image: Image.Image,
        target_class: str = "water",
    ) -> SegmentationResult:
        registry = get_model_registry()
        model = registry.get_model("SEGMENTATION")
        return model.segment(image, target_class=target_class)

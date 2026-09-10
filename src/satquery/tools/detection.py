from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.detection import DetectionResult
from satquery.schemas.models import ModelProvenance
from satquery.tools.base import BaseTool


class DetectionTool(BaseTool):
    """Tool for object instance detection and exact object counting in remote sensing imagery."""

    def __init__(self):
        super().__init__(
            name="DetectionTool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["detection", "detection_count"],
            provenance=ModelProvenance.HEURISTIC,
            description="Agentic tool detecting and counting discrete remote sensing objects without hallucination",
        )

    def execute(
        self,
        image: Image.Image,
        target_class: str = "building",
    ) -> DetectionResult:
        registry = get_model_registry()
        model = registry.get_model("DETECTION")
        return model.detect(image, target_class=target_class)

from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.grounding import GroundingResult
from satquery.schemas.models import ModelProvenance
from satquery.tools.base import BaseTool


class GroundingTool(BaseTool):
    """Tool for visual grounding and spatial bounding box extraction on satellite imagery."""

    def __init__(self):
        super().__init__(
            name="RemoteSensingGroundingTool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["grounding", "spatial_localization"],
            provenance=ModelProvenance.HEURISTIC,
            description="Agentic tool localizing natural language referring expressions with bounding boxes and masks",
        )

    def execute(
        self,
        image: Image.Image,
        query: str,
        target: Optional[str] = None,
    ) -> GroundingResult:
        registry = get_model_registry()
        model = registry.get_model("GROUNDING")
        return model.ground(image, query, target=target)

from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.models import ModelProvenance
from satquery.schemas.vqa import VQAAnswerResult
from satquery.tools.base import BaseTool


class CaptioningTool(BaseTool):
    """Tool for generating qualitative remote sensing scene descriptions."""

    def __init__(self):
        super().__init__(
            name="CaptioningTool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["scene_description", "captioning"],
            provenance=ModelProvenance.REAL,
            description="Agentic tool generating holistic semantic land-cover captions",
        )

    def execute(
        self,
        image: Image.Image,
        prompt: str = "Describe the land cover and major objects visible in this image.",
        context: Optional[Dict[str, Any]] = None,
    ) -> VQAAnswerResult:
        registry = get_model_registry()
        model = registry.get_model("VQA")
        return model.answer_question(image, prompt, context=context)

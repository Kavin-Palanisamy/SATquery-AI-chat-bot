from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.models import ModelProvenance
from satquery.schemas.vqa import VQAAnswerResult
from satquery.tools.base import BaseTool


class RemoteSensingVQATool(BaseTool):
    """Tool for answering multimodal natural language questions over satellite imagery."""

    def __init__(self):
        super().__init__(
            name="RemoteSensingVQATool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["vqa", "scene_description"],
            provenance=ModelProvenance.REAL,
            description="Agentic tool executing VQA and semantic reasoning over Earth Observation imagery",
        )

    def execute(
        self,
        image: Image.Image,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VQAAnswerResult:
        registry = get_model_registry()
        model = registry.get_model("VQA")
        if not model:
            from satquery.models.qwen_rs_vlm import RemoteSensingVLM
            model = RemoteSensingVLM(force_mock=True)
        return model.answer_question(image, question, context=context)

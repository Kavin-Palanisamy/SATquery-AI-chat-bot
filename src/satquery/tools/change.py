from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.change import ChangeResult
from satquery.schemas.models import ModelProvenance
from satquery.tools.base import BaseTool


class ChangeDetectionTool(BaseTool):
    """Tool for bi-temporal change detection and Change-VQA."""

    def __init__(self):
        super().__init__(
            name="RemoteSensingChangeTool",
            version="v3",
            supported_modalities=["OPTICAL", "MULTISPECTRAL"],
            supported_tasks=["bitemporal_change", "change_analysis"],
            provenance=ModelProvenance.REAL,
            description="Agentic tool conducting bi-temporal spectral change detection between two observation dates",
        )

    def execute(
        self,
        img_t1: Image.Image,
        img_t2: Image.Image,
        query: str,
        meta_t1: Optional[Dict] = None,
        meta_t2: Optional[Dict] = None,
    ) -> ChangeResult:
        registry = get_model_registry()
        model = registry.get_model("CHANGE")
        return model.analyze(img_t1, img_t2, query, meta_t1=meta_t1, meta_t2=meta_t2)
